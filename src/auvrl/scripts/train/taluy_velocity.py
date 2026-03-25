"""Train PPO on the Taluy 6D velocity-tracking task.

This is the first local training entrypoint for the new body-wrench velocity
task. Defaults are intentionally modest so you can start with a short smoke run
and scale up once the learning curves look sane.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
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
    make_taluy_velocity_env_cfg,
    taluy_velocity_ppo_runner_cfg,
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
        default="smoke",
        help="Optional suffix for the timestamped run directory.",
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
        default=20.0,
        help="Episode horizon in seconds.",
    )
    parser.add_argument(
        "--thruster-voltage-event-mode",
        "--thruster-voltage-randomization-mode",
        dest="thruster_voltage_event_mode",
        choices=("disabled", "startup", "reset"),
        default="disabled",
        help="Event timing for thruster supply voltage.",
    )
    parser.add_argument(
        "--current-event-mode",
        "--current-randomization-mode",
        dest="current_event_mode",
        choices=("disabled", "startup", "reset"),
        default="disabled",
        help="Event timing for water current.",
    )
    parser.add_argument(
        "--upload-model",
        action="store_true",
        help="Upload checkpoint files when using wandb logger.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _default_num_envs(device: str) -> int:
    return 256 if device.startswith("cuda") else 16


def _make_log_dir(experiment_name: str, run_name: str) -> Path:
    log_root = Path("logs") / "rsl_rl" / experiment_name
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    folder_name = stamp if not run_name else f"{stamp}_{run_name}"
    log_dir = log_root / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    num_envs = args.num_envs if args.num_envs is not None else _default_num_envs(device)

    if num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.iterations <= 0:
        raise SystemExit("--iterations must be positive.")

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    env_cfg = make_taluy_velocity_env_cfg(
        num_envs=num_envs,
        episode_length_s=args.episode_length_s,
        thruster_voltage_event_mode=args.thruster_voltage_event_mode,
        current_event_mode=args.current_event_mode,
    )
    env_cfg.seed = args.seed

    agent_cfg = taluy_velocity_ppo_runner_cfg()
    agent_cfg.seed = args.seed
    agent_cfg.max_iterations = args.iterations
    agent_cfg.run_name = args.run_name
    agent_cfg.upload_model = args.upload_model
    if args.num_steps_per_env is not None:
        if args.num_steps_per_env <= 0:
            raise SystemExit("--num-steps-per-env must be positive.")
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.experiment_name is not None:
        agent_cfg.experiment_name = args.experiment_name
    if args.logger is not None:
        agent_cfg.logger = args.logger
    if args.clip_actions is not None:
        agent_cfg.clip_actions = args.clip_actions

    log_dir = _make_log_dir(agent_cfg.experiment_name, agent_cfg.run_name)
    env_dict = cast(dict[str, Any], _yaml_safe(asdict(env_cfg)))
    agent_dict = cast(dict[str, Any], _yaml_safe(asdict(agent_cfg)))
    dump_yaml(log_dir / "params" / "env.yaml", env_dict)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_dict)

    print("Starting Taluy MJLab PPO training")
    print(
        f"device={device} num_envs={num_envs} iterations={agent_cfg.max_iterations} "
        f"num_steps_per_env={agent_cfg.num_steps_per_env}"
    )
    print(
        "events: "
        f"voltage={args.thruster_voltage_event_mode} "
        f"current={args.current_event_mode}"
    )
    print(f"log_dir={log_dir}")

    vec_env: RslRlVecEnvWrapper | None = None
    try:
        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = MjlabOnPolicyRunner(vec_env, asdict(agent_cfg), str(log_dir), device)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations,
            init_at_random_ep_len=True,
        )
    finally:
        if vec_env is not None:
            vec_env.close()


if __name__ == "__main__":
    main()
