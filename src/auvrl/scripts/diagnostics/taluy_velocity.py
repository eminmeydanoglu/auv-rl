"""Diagnose a trained Taluy 6D velocity policy with fixed-command sweeps.

This script runs a small bank of fixed body-velocity commands against a saved
checkpoint, reports steady-state tracking metrics, and prints a compact
TensorBoard scalar summary for the same run.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import os
from pathlib import Path
from typing import Any

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.os import get_checkpoint_path  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab RL dependencies. Ensure mjlab and rsl_rl are available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    UniformBodyVelocityCommand,
    make_taluy_velocity_env_cfg,
    taluy_velocity_ppo_runner_cfg,
)
from auvrl.actuator.body_wrench_action import BodyWrenchAction  # noqa: E402  # type: ignore[import-not-found]


CASES: tuple[tuple[str, tuple[float, float, float, float, float, float]], ...] = (
    ("hover", (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("surge+", (0.5, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("surge-", (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0)),
    ("sway+", (0.0, 0.5, 0.0, 0.0, 0.0, 0.0)),
    ("heave+", (0.0, 0.0, 0.3, 0.0, 0.0, 0.0)),
    ("roll+", (0.0, 0.0, 0.0, 0.6, 0.0, 0.0)),
    ("pitch+", (0.0, 0.0, 0.0, 0.0, 0.6, 0.0)),
    ("yaw+", (0.0, 0.0, 0.0, 0.0, 0.0, 0.8)),
)

TB_TAGS: tuple[str, ...] = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/track_angular_velocity",
    "Episode_Reward/action_l2",
    "Episode_Reward/action_rate_l2",
    "Episode_Reward/saturation",
    "Metrics/body_velocity/error_lin_vel",
    "Metrics/body_velocity/error_ang_vel",
    "Policy/mean_std",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Direct path to a checkpoint file. Overrides experiment/run lookup.",
    )
    parser.add_argument(
        "--experiment-name",
        default=taluy_velocity_ppo_runner_cfg().experiment_name,
        help="Experiment folder under logs/rsl_rl/ when --checkpoint-file is omitted.",
    )
    parser.add_argument(
        "--run-dir",
        default=".*",
        help="Regex for the run directory when resolving from logs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_.*.pt",
        help="Regex for the checkpoint filename when resolving from logs.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="Total rollout steps for each fixed-command case.",
    )
    parser.add_argument(
        "--tail-steps",
        type=int,
        default=400,
        help="Number of final steps used for steady-state metrics.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=300.0,
        help="Episode horizon during diagnosis.",
    )
    parser.add_argument(
        "--skip-tensorboard",
        action="store_true",
        help="Skip the TensorBoard scalar summary.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_file is not None:
        checkpoint_path = Path(args.checkpoint_file)
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path.resolve()

    log_root = ROOT / "logs" / "rsl_rl" / str(args.experiment_name)
    try:
        return get_checkpoint_path(log_root, args.run_dir, args.checkpoint)
    except Exception as exc:
        raise SystemExit(
            f"Failed to resolve checkpoint from {log_root}: {exc}"
        ) from exc


def _load_agent_cfg_dict(checkpoint_path: Path) -> dict[str, Any]:
    params_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not params_path.exists():
        return asdict(taluy_velocity_ppo_runner_cfg())

    with params_path.open("r", encoding="utf-8") as file:
        cfg = yaml.full_load(file)
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"Expected mapping in {params_path}, got {type(cfg).__name__}."
        )
    return cfg


def _format_vec(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(v):+.3f}" for v in values.tolist()) + "]"


def _build_eval_env(device: str, episode_length_s: float) -> ManagerBasedRlEnv:
    cfg = make_taluy_velocity_env_cfg(
        num_envs=len(CASES),
        episode_length_s=episode_length_s,
        command_resampling_time_s=(1.0e6, 1.0e6),
        command_rel_zero_envs=1.0,
        command_init_velocity_prob=0.0,
    )
    cfg.terminations = {}
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def _set_fixed_commands(base_env: ManagerBasedRlEnv) -> torch.Tensor:
    term = base_env.command_manager.get_term("body_velocity")
    if not isinstance(term, UniformBodyVelocityCommand):
        raise RuntimeError("Taluy body_velocity command term is not available.")

    command = torch.tensor(
        [values for _, values in CASES],
        device=base_env.device,
        dtype=torch.float,
    )
    term.vel_command_b[:] = command
    term.is_zero_env[:] = torch.linalg.vector_norm(command, dim=1) <= 1.0e-6
    term.time_left[:] = 1.0e9
    return command


def _run_fixed_command_sweep(
    checkpoint_path: Path,
    *,
    device: str,
    steps: int,
    tail_steps: int,
    episode_length_s: float,
) -> None:
    if steps <= 0:
        raise SystemExit("--steps must be positive.")
    if tail_steps <= 0:
        raise SystemExit("--tail-steps must be positive.")
    if tail_steps > steps:
        raise SystemExit("--tail-steps cannot exceed --steps.")
    if episode_length_s <= 0.0:
        raise SystemExit("--episode-length-s must be positive.")

    agent_cfg_dict = _load_agent_cfg_dict(checkpoint_path)
    base_env = _build_eval_env(device, episode_length_s)
    env = RslRlVecEnvWrapper(
        env=base_env, clip_actions=agent_cfg_dict.get("clip_actions")
    )

    try:
        runner = MjlabOnPolicyRunner(env, agent_cfg_dict, device=device)
        runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        env.reset()
        command = _set_fixed_commands(base_env)
        obs = env.get_observations()

        robot = base_env.scene["robot"]
        wrench_term = base_env.action_manager.get_term("body_wrench")
        if not isinstance(wrench_term, BodyWrenchAction):
            raise RuntimeError("Taluy body_wrench action term is not available.")

        num_thrusters = wrench_term.thruster_targets.shape[1]
        allocation_t = wrench_term.allocation_matrix_b.transpose(0, 1)

        mean_lin_sum = torch.zeros((len(CASES), 3), device=base_env.device)
        mean_ang_sum = torch.zeros((len(CASES), 3), device=base_env.device)
        mean_policy_action_raw_sum = torch.zeros(
            (len(CASES), 6), device=base_env.device
        )
        mean_effective_action_sum = torch.zeros((len(CASES), 6), device=base_env.device)
        mean_desired_wrench_sum = torch.zeros((len(CASES), 6), device=base_env.device)
        mean_applied_wrench_sum = torch.zeros((len(CASES), 6), device=base_env.device)
        mean_allocated_wrench_sum = torch.zeros((len(CASES), 6), device=base_env.device)
        mean_alloc_error_sum = torch.zeros((len(CASES), 6), device=base_env.device)
        mean_thruster_targets_sum = torch.zeros(
            (len(CASES), num_thrusters), device=base_env.device
        )
        mean_abs_thruster_targets_sum = torch.zeros(
            (len(CASES), num_thrusters), device=base_env.device
        )
        lin_err_sq_sum = torch.zeros(len(CASES), device=base_env.device)
        ang_err_sq_sum = torch.zeros(len(CASES), device=base_env.device)
        alloc_error_sq_sum = torch.zeros(len(CASES), device=base_env.device)
        thr_max_sum = torch.zeros(len(CASES), device=base_env.device)
        sat_sum = torch.zeros(len(CASES), device=base_env.device)

        for step in range(steps):
            with torch.no_grad():
                action = policy(obs)
            obs, _, _, _ = env.step(action)

            if step < steps - tail_steps:
                continue

            lin_vel = robot.data.root_link_lin_vel_b
            ang_vel = robot.data.root_link_ang_vel_b
            effective_action = wrench_term.raw_action
            desired_wrench = wrench_term.desired_wrench_b
            applied_wrench = wrench_term.applied_wrench_origin_b
            allocated_wrench = wrench_term.thruster_targets @ allocation_t
            alloc_error = allocated_wrench - applied_wrench
            mean_lin_sum += lin_vel
            mean_ang_sum += ang_vel
            mean_policy_action_raw_sum += action
            mean_effective_action_sum += effective_action
            mean_desired_wrench_sum += desired_wrench
            mean_applied_wrench_sum += applied_wrench
            mean_allocated_wrench_sum += allocated_wrench
            mean_alloc_error_sum += alloc_error
            mean_thruster_targets_sum += wrench_term.thruster_targets
            mean_abs_thruster_targets_sum += wrench_term.thruster_targets.abs()
            lin_err_sq_sum += torch.sum(torch.square(command[:, :3] - lin_vel), dim=1)
            ang_err_sq_sum += torch.sum(torch.square(command[:, 3:] - ang_vel), dim=1)
            alloc_error_sq_sum += torch.sum(torch.square(alloc_error), dim=1)
            thr_max_sum += wrench_term.thruster_targets.abs().amax(dim=1)
            sat_sum += wrench_term.step_saturation_fraction

        mean_lin = mean_lin_sum / float(tail_steps)
        mean_ang = mean_ang_sum / float(tail_steps)
        mean_policy_action_raw = mean_policy_action_raw_sum / float(tail_steps)
        mean_effective_action = mean_effective_action_sum / float(tail_steps)
        mean_desired_wrench = mean_desired_wrench_sum / float(tail_steps)
        mean_applied_wrench = mean_applied_wrench_sum / float(tail_steps)
        mean_allocated_wrench = mean_allocated_wrench_sum / float(tail_steps)
        mean_alloc_error = mean_alloc_error_sum / float(tail_steps)
        mean_thruster_targets = mean_thruster_targets_sum / float(tail_steps)
        mean_abs_thruster_targets = mean_abs_thruster_targets_sum / float(tail_steps)
        rms_lin = torch.sqrt(lin_err_sq_sum / float(tail_steps))
        rms_ang = torch.sqrt(ang_err_sq_sum / float(tail_steps))
        alloc_error_rms = torch.sqrt(alloc_error_sq_sum / float(tail_steps))
        mean_thr_max = thr_max_sum / float(tail_steps)
        mean_sat = sat_sum / float(tail_steps)

        print("Fixed-command sweep")
        print(f"checkpoint={checkpoint_path}")
        print(f"device={device} steps={steps} tail_steps={tail_steps}")
        print()
        for index, (name, _) in enumerate(CASES):
            print(f"{name}")
            print(f"  cmd={_format_vec(command[index])}")
            print(f"  mean_lin={_format_vec(mean_lin[index])}")
            print(f"  mean_ang={_format_vec(mean_ang[index])}")
            print(
                f"  mean_policy_action_raw={_format_vec(mean_policy_action_raw[index])}"
            )
            print(
                f"  mean_effective_action={_format_vec(mean_effective_action[index])}"
            )
            print(f"  mean_desired_wrench={_format_vec(mean_desired_wrench[index])}")
            print(f"  mean_applied_wrench={_format_vec(mean_applied_wrench[index])}")
            print(
                f"  mean_allocated_wrench={_format_vec(mean_allocated_wrench[index])}"
            )
            print(f"  mean_alloc_error={_format_vec(mean_alloc_error[index])}")
            print(
                f"  mean_thruster_targets={_format_vec(mean_thruster_targets[index])}"
            )
            print(
                "  "
                f"mean_abs_thruster_targets={_format_vec(mean_abs_thruster_targets[index])}"
            )
            print(
                "  "
                f"rms_lin={float(rms_lin[index]):.3f} "
                f"rms_ang={float(rms_ang[index]):.3f} "
                f"alloc_err_rms={float(alloc_error_rms[index]):.3f} "
                f"thr_max={float(mean_thr_max[index]):.3f} "
                f"sat={float(mean_sat[index]):.3f}"
            )
            print()
    finally:
        env.close()


def _print_tensorboard_summary(checkpoint_path: Path) -> None:
    try:
        from tensorboard.backend.event_processing import (  # type: ignore[import-not-found]
            event_accumulator,
        )
    except ModuleNotFoundError:
        print("TensorBoard summary")
        print("  skipped: tensorboard is not installed")
        print()
        return

    run_dir = checkpoint_path.parent
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        print("TensorBoard summary")
        print(f"  skipped: no event file found in {run_dir}")
        print()
        return

    event_file = event_files[-1]
    accumulator = event_accumulator.EventAccumulator(str(event_file))
    accumulator.Reload()
    scalar_tags = set(accumulator.Tags().get("scalars", []))

    print("TensorBoard summary")
    print(f"event_file={event_file}")
    for tag in TB_TAGS:
        if tag not in scalar_tags:
            print(f"  {tag}: missing")
            continue
        series = accumulator.Scalars(tag)
        if not series:
            print(f"  {tag}: empty")
            continue
        values = [float(item.value) for item in series]
        max_index = max(range(len(values)), key=values.__getitem__)
        min_index = min(range(len(values)), key=values.__getitem__)
        print(
            "  "
            f"{tag}: first={values[0]:.6f} last={values[-1]:.6f} "
            f"min={values[min_index]:.6f}@{series[min_index].step} "
            f"max={values[max_index]:.6f}@{series[max_index].step}"
        )
    print()


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint_path(args)

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")

    _run_fixed_command_sweep(
        checkpoint_path,
        device=device,
        steps=args.steps,
        tail_steps=args.tail_steps,
        episode_length_s=args.episode_length_s,
    )
    if not args.skip_tensorboard:
        _print_tensorboard_summary(checkpoint_path)


if __name__ == "__main__":
    main()
