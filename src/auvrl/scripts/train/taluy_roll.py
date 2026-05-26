"""Train PPO on the deterministic Taluy roll task."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Any, cast

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.os import dump_yaml  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab RL dependencies. Ensure mjlab and rsl_rl are available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    ROLL_CURRICULUM_STAGES,
    get_roll_curriculum_stage,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.tasks.roll.auto_curriculum import (  # noqa: E402
    POST_C3L_POLISH_AUTO_CURRICULUM,
)
from auvrl.tasks.roll.eval_rules import (  # noqa: E402
    capture_roll_eval_rules,
    write_roll_eval_rules,
)


def _yaml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _yaml_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_yaml_safe(item) for item in value]
    if isinstance(value, list):
        return [_yaml_safe(item) for item in value]
    return value


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Training device. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Parallel env count. Default: 256 on CUDA, 16 on CPU.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="PPO iterations. Default is a short smoke-scale run.",
    )
    parser.add_argument(
        "--num-steps-per-env",
        type=int,
        default=None,
        help="Rollout steps per env per PPO update. Defaults to the runner config.",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="Checkpoint save interval in PPO iterations. Defaults to the runner config.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override PPO learning rate.",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=None,
        help="Override PPO entropy coefficient.",
    )
    parser.add_argument(
        "--desired-kl",
        type=float,
        default=None,
        help="Override PPO adaptive-schedule desired KL.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Training seed.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment folder under logs/rsl_rl/. Defaults to the runner config.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix for the timestamped run directory.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default=None,
        help="Static roll curriculum stage to train. Example: c0_90_discovery.",
    )
    parser.add_argument(
        "--auto-curriculum",
        choices=(POST_C3L_POLISH_AUTO_CURRICULUM,),
        default=None,
        help="Enable an adaptive roll auto-curriculum.",
    )
    parser.add_argument(
        "--auto-curriculum-goal-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default="c3q_720_c3l_strict_settle",
        help="Goal reference stage for --auto-curriculum.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        default=None,
        help="Optional PPO checkpoint to load before continuing this stage.",
    )
    parser.add_argument(
        "--resume-mode",
        choices=("weights-only", "full"),
        default="weights-only",
        help=(
            "How to load --resume-checkpoint. 'weights-only' warm-starts actor/critic "
            "with a fresh optimizer/LR; 'full' restores optimizer, iteration, and env state."
        ),
    )
    parser.add_argument(
        "--logger",
        choices=("tensorboard", "wandb"),
        default=None,
        help="Training logger backend. Defaults to the runner config.",
    )
    parser.add_argument(
        "--clip-actions",
        type=float,
        default=None,
        help="Optional scalar action clip before env.step(). Default leaves it disabled.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Episode horizon in seconds. Defaults to the selected stage or 20s.",
    )
    parser.add_argument(
        "--upload-model",
        action="store_true",
        help="Upload checkpoint files when using wandb logger.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="Run roll eval every N PPO updates during training.",
    )
    parser.add_argument(
        "--eval-num-envs",
        type=int,
        default=256,
        help="Parallel env count for in-training eval.",
    )
    parser.add_argument(
        "--eval-max-steps",
        type=int,
        default=None,
        help="Maximum env steps per in-training eval episode.",
    )
    parser.add_argument(
        "--eval-suite",
        default=None,
        help="Eval suite name under logs/tensorboard_eval/roll. Defaults to the training run folder.",
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=ROOT / "logs" / "tensorboard_eval" / "roll",
        help="Root directory for in-training eval artifacts.",
    )
    parser.add_argument(
        "--eval-device",
        choices=("same", "auto", "cpu", "cuda"),
        default="same",
        help="Device for in-training eval. 'same' reuses the training device.",
    )
    parser.add_argument(
        "--eval-curriculum-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default=None,
        help="Curriculum stage used by in-training eval. Defaults to --curriculum-stage.",
    )
    parser.add_argument(
        "--eval-skip-images",
        action="store_true",
        help="Skip PNG dashboards during in-training eval.",
    )
    parser.add_argument(
        "--no-eval-final",
        action="store_true",
        help="Disable final checkpoint eval when --eval-interval is set.",
    )
    parser.add_argument(
        "--eval-continue-on-error",
        action="store_true",
        help="Continue training if an in-training eval fails.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    # Honor torchrun-provided LOCAL_RANK so each rank pins itself to its own GPU.
    local_rank_env = os.environ.get("LOCAL_RANK")
    if local_rank_env is not None and torch.cuda.is_available():
        local_rank = int(local_rank_env)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(local_rank))
        return f"cuda:{local_rank}"
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _default_num_envs(device: str) -> int:
    return 256 if device.startswith("cuda") else 16


def _make_log_dir(experiment_name: str, run_name: str) -> Path:
    # In multi-rank torchrun, all ranks must share the SAME log_dir. Allow the
    # caller (sbatch / launcher) to pin it via AUVRL_LOG_DIR.
    override = os.environ.get("AUVRL_LOG_DIR")
    if override:
        log_dir = Path(override)
    else:
        log_root = Path("logs") / "rsl_rl" / experiment_name
        stamp = os.environ.get("AUVRL_LOG_STAMP") or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = stamp if not run_name else f"{stamp}_{run_name}"
        log_dir = log_root / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _validate_eval_args(args: argparse.Namespace) -> None:
    if args.eval_interval is not None and args.eval_interval <= 0:
        raise SystemExit("--eval-interval must be positive.")
    if args.eval_num_envs <= 0:
        raise SystemExit("--eval-num-envs must be positive.")
    if args.eval_max_steps is not None and args.eval_max_steps <= 0:
        raise SystemExit("--eval-max-steps must be positive when set.")


def _eval_device_arg(eval_device: str, train_device: str) -> str:
    if eval_device != "same":
        return eval_device
    return train_device


def _goal_eval_stage(args: argparse.Namespace) -> str | None:
    if args.eval_curriculum_stage is not None:
        return args.eval_curriculum_stage
    if args.auto_curriculum is not None:
        return args.auto_curriculum_goal_stage
    return args.curriculum_stage


def _summary_value(summary: dict[str, Any], *path: str) -> float | None:
    value: Any = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    if isinstance(value, int | float):
        return float(value)
    return None


def _log_eval_summary_to_training_tb(
    summary: dict[str, Any],
    log_dir: Path,
    iteration: int,
    writer: Any | None,
    *,
    tag_prefix: str = "EvalInTraining",
) -> None:
    created_writer = None
    target_writer = writer
    if target_writer is None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ModuleNotFoundError:
            return
        created_writer = SummaryWriter(log_dir=str(log_dir))
        target_writer = created_writer
    try:
        scalar_paths = {
            f"{tag_prefix}/done_rate": ("done_rate",),
            f"{tag_prefix}/success_rate": ("outcome", "success_rate"),
            f"{tag_prefix}/task_success_rate": ("outcome", "task_success_rate"),
            f"{tag_prefix}/first_done_time_s_mean": ("first_done_time_s", "mean"),
            f"{tag_prefix}/xy_drift_peak_mean": ("terminal", "final_xy_drift_m_peak", "mean"),
            f"{tag_prefix}/xy_drift_time_mean": ("trajectory", "xy_drift_m", "time_mean_of_env_mean"),
            f"{tag_prefix}/saturation_time_mean": (
                "trajectory",
                "body_wrench_saturation_fraction",
                "time_mean_of_env_mean",
            ),
        }
        for tag, path in scalar_paths.items():
            value = _summary_value(summary, *path)
            if value is not None and math.isfinite(value):
                target_writer.add_scalar(tag, value, iteration)
    finally:
        if created_writer is not None:
            created_writer.flush()
            created_writer.close()


def _run_checkpoint_eval(
    *,
    checkpoint_path: Path,
    iteration: int,
    args: argparse.Namespace,
    train_device: str,
    log_dir: Path,
    writer: Any | None,
    eval_kind: str,
    curriculum_stage: str | None,
    eval_rules: dict[str, Any] | None = None,
) -> None:
    from auvrl.scripts.eval import taluy_roll_tensorboard as roll_eval
    from auvrl.scripts.eval.roll_eval_leaderboard import write_leaderboard

    suite = roll_eval._safe_label(args.eval_suite or log_dir.name)
    suite_dir = args.eval_output_root.expanduser().resolve() / suite
    safe_kind = roll_eval._safe_label(eval_kind)
    label = roll_eval._safe_label(f"{log_dir.name}_{safe_kind}_model_{iteration}")
    run_dir = suite_dir / label
    rules_path = None
    if eval_rules is not None:
        rules_path = run_dir / "eval_rules.json"
        write_roll_eval_rules(rules_path, eval_rules)
    eval_args = argparse.Namespace(
        device=_eval_device_arg(args.eval_device, train_device),
        num_envs=args.eval_num_envs,
        max_steps=args.eval_max_steps,
        curriculum_stage=curriculum_stage,
        eval_rules_path=rules_path,
        eval_kind=safe_kind,
        episode_length_s=args.episode_length_s,
        roll_direction=1,
    )
    roll_eval._ensure_output_dir(run_dir, overwrite=True)
    distributed_env_keys = ("WORLD_SIZE", "RANK", "LOCAL_RANK", "LOCAL_WORLD_SIZE")
    distributed_env = {key: os.environ.get(key) for key in distributed_env_keys}
    for key in distributed_env_keys:
        os.environ.pop(key, None)
    try:
        series = roll_eval.evaluate_checkpoint(checkpoint_path, eval_args, label=label)
    finally:
        for key, value in distributed_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    summary = roll_eval.write_artifacts(series, run_dir, skip_images=args.eval_skip_images)
    write_leaderboard(
        search_root=suite_dir,
        csv_path=suite_dir / "leaderboard.csv",
        markdown_path=suite_dir / "leaderboard.md",
    )
    _log_eval_summary_to_training_tb(
        summary,
        log_dir,
        iteration,
        writer,
        tag_prefix=f"EvalInTraining/{safe_kind}",
    )
    timing = summary["first_done_time_s"]
    success = summary.get("outcome", {}).get("success_rate")
    success_text = f"{success:.3f}" if isinstance(success, int | float) else "n/a"
    print(
        f"in_training_eval kind={safe_kind} iteration={iteration} label={label} "
        f"done={summary['done_count']}/{summary['num_envs']} "
        f"success_rate={success_text} "
        f"mean_first_done={timing['mean']:.3f}s "
        f"run_dir={run_dir}"
    )


def _install_eval_hook(
    runner: MjlabOnPolicyRunner,
    *,
    env: ManagerBasedRlEnv,
    log_dir: Path,
    args: argparse.Namespace,
    device: str,
) -> set[int]:
    evaluated_iterations: set[int] = set()
    original_log = runner.logger.log
    eval_rank = int(getattr(runner, "gpu_global_rank", 0))
    distributed = bool(getattr(runner, "is_distributed", False))

    def hooked_log(*log_args: Any, **log_kwargs: Any) -> Any:
        result = original_log(*log_args, **log_kwargs)
        iteration = log_kwargs.get("it")
        if iteration is None and log_args:
            iteration = log_args[0]
        if not isinstance(iteration, int):
            return result
        if (iteration + 1) % int(args.eval_interval) != 0:
            return result
        checkpoint_path = log_dir / f"model_{iteration}.pt"
        if distributed:
            torch.distributed.barrier()
        eval_error: Exception | None = None
        if eval_rank == 0:
            try:
                runner.save(str(checkpoint_path))
                current_stage = args.curriculum_stage
                if current_stage is None and args.auto_curriculum is not None:
                    current_stage = "c3l_720_xy_guard"
                if args.auto_curriculum is not None:
                    _run_checkpoint_eval(
                        checkpoint_path=checkpoint_path,
                        iteration=iteration,
                        args=args,
                        train_device=device,
                        log_dir=log_dir,
                        writer=runner.logger.writer,
                        eval_kind="current",
                        curriculum_stage=current_stage,
                        eval_rules=capture_roll_eval_rules(env, mode="current"),
                    )
                _run_checkpoint_eval(
                    checkpoint_path=checkpoint_path,
                    iteration=iteration,
                    args=args,
                    train_device=device,
                    log_dir=log_dir,
                    writer=runner.logger.writer,
                    eval_kind="goal",
                    curriculum_stage=_goal_eval_stage(args),
                )
                evaluated_iterations.add(iteration)
            except Exception as exc:
                eval_error = exc
                if args.eval_continue_on_error:
                    print(f"in_training_eval_failed iteration={iteration} error={exc}")
        if distributed:
            fail = torch.tensor(
                [1 if eval_error is not None else 0],
                device=torch.device(device if device.startswith("cuda") else "cpu"),
            )
            torch.distributed.broadcast(fail, src=0)
            if int(fail.item()) and not args.eval_continue_on_error and eval_rank != 0:
                raise RuntimeError(f"in-training eval failed on rank 0 at iteration {iteration}.")
        if eval_error is not None and not args.eval_continue_on_error:
            raise eval_error
        return result

    runner.logger.log = hooked_log
    return evaluated_iterations


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    num_envs = args.num_envs if args.num_envs is not None else _default_num_envs(device)
    _validate_eval_args(args)

    if num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive.")

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    # Rank-aware seed offset so different distributed workers see diverse rollouts.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    seed = args.seed + rank

    env_cfg = make_taluy_roll_env_cfg(
        num_envs=num_envs,
        curriculum_stage=args.curriculum_stage,
        episode_length_s=args.episode_length_s,
        auto_curriculum=args.auto_curriculum,
        auto_curriculum_goal_stage=args.auto_curriculum_goal_stage,
    )
    env_cfg.seed = seed

    agent_cfg = taluy_roll_ppo_runner_cfg()
    agent_cfg.seed = seed
    agent_cfg.max_iterations = args.iterations
    if args.run_name is not None:
        run_name = args.run_name
    elif args.auto_curriculum is not None:
        start_stage = args.curriculum_stage or "c3l_720_xy_guard"
        run_name = f"{args.auto_curriculum}_from_{start_stage}"
    else:
        run_name = args.curriculum_stage or "smoke"
    agent_cfg.run_name = run_name
    agent_cfg.upload_model = args.upload_model
    if args.num_steps_per_env is not None:
        if args.num_steps_per_env <= 0:
            raise SystemExit("--num-steps-per-env must be positive.")
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        if args.save_interval <= 0:
            raise SystemExit("--save-interval must be positive.")
        agent_cfg.save_interval = args.save_interval
    if args.learning_rate is not None:
        if args.learning_rate <= 0.0:
            raise SystemExit("--learning-rate must be positive.")
        agent_cfg.algorithm.learning_rate = args.learning_rate
    if args.entropy_coef is not None:
        if args.entropy_coef < 0.0:
            raise SystemExit("--entropy-coef must be non-negative.")
        agent_cfg.algorithm.entropy_coef = args.entropy_coef
    if args.desired_kl is not None:
        if args.desired_kl <= 0.0:
            raise SystemExit("--desired-kl must be positive.")
        agent_cfg.algorithm.desired_kl = args.desired_kl
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if args.clip_actions is not None:
        agent_cfg.clip_actions = args.clip_actions

    log_dir = _make_log_dir(agent_cfg.experiment_name, agent_cfg.run_name)
    is_rank0 = rank == 0
    if is_rank0:
        env_dict = cast(dict[str, Any], _yaml_safe(asdict(env_cfg)))
        agent_dict = cast(dict[str, Any], _yaml_safe(asdict(agent_cfg)))
        dump_yaml(log_dir / "params" / "env.yaml", env_dict)
        dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)

    if is_rank0:
        print("Starting Taluy MJLab PPO training")
        if world_size > 1:
            print(f"distributed=true world_size={world_size}")
    print(
        f"rank={rank}/{world_size} device={device} num_envs={num_envs} iterations={agent_cfg.max_iterations} "
        f"num_steps_per_env={agent_cfg.num_steps_per_env}"
    )
    print(
        f"save_interval={agent_cfg.save_interval} "
        f"learning_rate={agent_cfg.algorithm.learning_rate} "
        f"entropy_coef={agent_cfg.algorithm.entropy_coef} "
        f"desired_kl={agent_cfg.algorithm.desired_kl}"
    )
    printed_stage_name = args.curriculum_stage
    if printed_stage_name is None and args.auto_curriculum is not None:
        printed_stage_name = "c3l_720_xy_guard"
    if printed_stage_name is None:
        print("task=roll_v1 target_roll_deg=720.0 roll_direction=1 settle_window_s=1.0")
    else:
        stage = get_roll_curriculum_stage(printed_stage_name)
        print(
            f"task=roll_v1 curriculum_stage={stage.name} "
            f"target_roll_deg={stage.target_roll_deg} "
            f"episode_length_s={env_cfg.episode_length_s} "
            f"settle_window_s={stage.settle_window_s}"
        )
        print(f"stage_description={stage.description}")
    if args.auto_curriculum is not None:
        print(
            f"auto_curriculum={args.auto_curriculum} "
            f"auto_curriculum_goal_stage={args.auto_curriculum_goal_stage}"
        )
    if args.resume_checkpoint is not None:
        print(f"resume_checkpoint={args.resume_checkpoint}")
        print(f"resume_mode={args.resume_mode}")
    print(f"log_dir={log_dir}")

    vec_env: RslRlVecEnvWrapper | None = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), str(log_dir), device)
        evaluated_iterations: set[int] = set()
        if args.eval_interval is not None:
            evaluated_iterations = _install_eval_hook(
                runner,
                env=env,
                log_dir=log_dir,
                args=args,
                device=device,
            )
        if args.resume_checkpoint is not None:
            if not args.resume_checkpoint.exists():
                raise SystemExit(f"Checkpoint file not found: {args.resume_checkpoint}")
            if args.resume_mode == "weights-only":
                runner.load(
                    str(args.resume_checkpoint),
                    load_cfg={
                        "actor": True,
                        "critic": True,
                        "optimizer": False,
                        "iteration": False,
                        "rnd": False,
                    },
                    map_location=device,
                )
                env.unwrapped.common_step_counter = 0
            else:
                runner.load(str(args.resume_checkpoint), map_location=device)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
        final_iteration = int(runner.current_learning_iteration)
        if (
            args.eval_interval is not None
            and is_rank0
            and not args.no_eval_final
            and final_iteration not in evaluated_iterations
        ):
            checkpoint_path = log_dir / f"model_{final_iteration}.pt"
            try:
                runner.save(str(checkpoint_path))
                current_stage = args.curriculum_stage
                if current_stage is None and args.auto_curriculum is not None:
                    current_stage = "c3l_720_xy_guard"
                if args.auto_curriculum is not None:
                    _run_checkpoint_eval(
                        checkpoint_path=checkpoint_path,
                        iteration=final_iteration,
                        args=args,
                        train_device=device,
                        log_dir=log_dir,
                        writer=None,
                        eval_kind="current",
                        curriculum_stage=current_stage,
                        eval_rules=capture_roll_eval_rules(env, mode="current"),
                    )
                _run_checkpoint_eval(
                    checkpoint_path=checkpoint_path,
                    iteration=final_iteration,
                    args=args,
                    train_device=device,
                    log_dir=log_dir,
                    writer=None,
                    eval_kind="goal",
                    curriculum_stage=_goal_eval_stage(args),
                )
            except Exception as exc:
                if not args.eval_continue_on_error:
                    raise
                print(f"in_training_eval_failed iteration={final_iteration} error={exc}")
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main()
