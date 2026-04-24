"""Run a quick health probe for the deterministic Taluy roll task."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import cast

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab dependencies. Ensure mjlab is available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    ROLL_CURRICULUM_STAGES,
    get_roll_curriculum_stage,
    make_taluy_roll_env_cfg,
)
from auvrl.tasks.roll.runtime import (  # noqa: E402  # type: ignore[import-not-found]
    action_term_slice,
    get_roll_task_state,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=4,
        help="Parallel env count for the diagnostic rollout.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of rollout steps to execute after reset.",
    )
    parser.add_argument(
        "--roll-torque",
        type=float,
        default=0.20,
        help="Normalized body_wrench roll torque command used during the probe.",
    )
    parser.add_argument(
        "--target-roll-deg",
        type=float,
        default=720.0,
        help="Roll task target in degrees.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default=None,
        help="Static roll curriculum stage to probe. Overrides --target-roll-deg.",
    )
    parser.add_argument(
        "--roll-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Commanded roll direction.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Episode horizon in seconds. Defaults to the selected stage or 20s.",
    )
    parser.add_argument(
        "--settle-window-s",
        type=float,
        default=1.0,
        help="Success settle window in seconds.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _format_stats(values: torch.Tensor) -> str:
    values_cpu = values.detach().float().cpu()
    return (
        f"min={values_cpu.min().item():+.5f} "
        f"mean={values_cpu.mean().item():+.5f} "
        f"max={values_cpu.max().item():+.5f}"
    )


def _make_env(args: argparse.Namespace, device: str) -> ManagerBasedRlEnv:
    cfg = make_taluy_roll_env_cfg(
        num_envs=args.num_envs,
        curriculum_stage=args.curriculum_stage,
        target_roll_deg=args.target_roll_deg,
        roll_direction=args.roll_direction,
        episode_length_s=args.episode_length_s,
        settle_window_s=args.settle_window_s,
    )
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)

    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.steps < 0:
        raise SystemExit("--steps must be non-negative.")
    if not -1.0 <= args.roll_torque <= 1.0:
        raise SystemExit("--roll-torque must be in [-1, 1].")

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    target_roll_deg = args.target_roll_deg
    if args.curriculum_stage is not None:
        target_roll_deg = get_roll_curriculum_stage(args.curriculum_stage).target_roll_deg

    env = _make_env(args, device)
    try:
        obs_raw, _ = env.reset()
        obs = cast(dict[str, torch.Tensor], obs_raw)
        action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
        )
        body_wrench_slice = action_term_slice(env, "body_wrench")
        action[:, body_wrench_slice.start + 3] = float(args.roll_torque)

        reward_sum = torch.zeros(env.num_envs, device=env.device)
        terminated_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        truncated_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        last_reward = torch.zeros(env.num_envs, device=env.device)

        for _ in range(args.steps):
            obs_raw, last_reward, terminated, truncated, _extras = env.step(action)
            obs = cast(dict[str, torch.Tensor], obs_raw)
            reward_sum += last_reward
            terminated_count += terminated.long()
            truncated_count += truncated.long()

        state = get_roll_task_state(env)
        target_roll_rad = math.radians(target_roll_deg)
        signed_progress = float(args.roll_direction) * state.phi_total_rad
        progress_ratio = signed_progress / target_roll_rad

        print("Taluy roll diagnostics passed.")
        print(f"  device: {device}")
        print(f"  num_envs: {env.num_envs}")
        print(f"  steps: {args.steps}")
        print(f"  curriculum_stage: {args.curriculum_stage}")
        print(f"  target_roll_deg: {target_roll_deg}")
        print(f"  actor_obs_shape: {tuple(obs['actor'].shape)}")
        print(f"  critic_obs_shape: {tuple(obs['critic'].shape)}")
        print(f"  action_terms: {list(env.action_manager.active_terms)}")
        print(f"  reward_terms: {list(env.reward_manager.active_terms)}")
        print(f"  termination_terms: {list(env.termination_manager.active_terms)}")
        print(f"  phi_total_rad: {_format_stats(state.phi_total_rad)}")
        print(f"  delta_roll_rad: {_format_stats(state.delta_roll_rad)}")
        print(f"  progress_ratio: {_format_stats(progress_ratio)}")
        print(f"  last_reward: {_format_stats(last_reward)}")
        print(f"  reward_sum: {_format_stats(reward_sum)}")
        print(f"  terminated_count: {terminated_count.detach().cpu().tolist()}")
        print(f"  truncated_count: {truncated_count.detach().cpu().tolist()}")
        print(
            "  target_reached: "
            f"{state.target_reached.detach().cpu().tolist()}"
        )
        print(
            "  settle_counter_steps: "
            f"{state.settle_counter_steps.detach().cpu().tolist()}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
