"""Integration smoke tests for the Taluy MJLab thruster environment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. Ensure mjlab is available (local ../mjlab or installed dependency)."
    ) from exc

from auvrl import make_taluy_base_env_cfg  # type: ignore[import-not-found]  # noqa: E402


@dataclass
class RolloutStats:
    max_lin_speed: float
    max_ang_speed: float
    max_surge_vel: float
    min_yaw_vel: float
    final_surge_vel: float
    final_yaw_vel: float


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env(decimation: int, physics_dt_s: float = 0.002) -> ManagerBasedRlEnv:
    cfg = make_taluy_base_env_cfg(
        action_space="thruster",
    )
    cfg.decimation = decimation
    cfg.sim.mujoco.timestep = physics_dt_s
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _rollout(
    env: ManagerBasedRlEnv,
    *,
    total_time_s: float,
    pulse_thruster_idx: int | None,
    pulse_command: float,
    pulse_duration_s: float,
) -> RolloutStats:
    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )

    _, _ = env.reset()

    num_steps = max(int(total_time_s / env.step_dt), 1)
    max_lin_speed = 0.0
    max_ang_speed = 0.0
    max_surge_vel = 0.0
    min_yaw_vel = 0.0

    for step in range(num_steps):
        sim_time = step * env.step_dt
        action.zero_()
        if pulse_thruster_idx is not None and sim_time < pulse_duration_s:
            action[:, pulse_thruster_idx] = pulse_command

        env.step(action)
        _assert_finite_env_state(env)

        robot = env.scene["robot"]
        lin_vel_b = robot.data.root_link_lin_vel_b[0]
        ang_vel_b = robot.data.root_link_ang_vel_b[0]

        lin_speed = float(torch.linalg.norm(lin_vel_b).item())
        ang_speed = float(torch.linalg.norm(ang_vel_b).item())
        surge_vel = float(lin_vel_b[0].item())
        yaw_vel = float(ang_vel_b[2].item())

        max_lin_speed = max(max_lin_speed, lin_speed)
        max_ang_speed = max(max_ang_speed, ang_speed)
        max_surge_vel = max(max_surge_vel, surge_vel)
        min_yaw_vel = min(min_yaw_vel, yaw_vel)

    final_robot = env.scene["robot"]
    final_lin = final_robot.data.root_link_lin_vel_b[0]
    final_ang = final_robot.data.root_link_ang_vel_b[0]
    return RolloutStats(
        max_lin_speed=max_lin_speed,
        max_ang_speed=max_ang_speed,
        max_surge_vel=max_surge_vel,
        min_yaw_vel=min_yaw_vel,
        final_surge_vel=float(final_lin[0].item()),
        final_yaw_vel=float(final_ang[2].item()),
    )


def _check_zero_action_trim() -> None:
    env = _make_env(decimation=4)
    try:
        stats = _rollout(
            env,
            total_time_s=1.0,
            pulse_thruster_idx=None,
            pulse_command=0.0,
            pulse_duration_s=0.0,
        )
    finally:
        env.close()

    if stats.max_lin_speed > 0.05:
        raise AssertionError(
            f"Zero-command linear drift too high: {stats.max_lin_speed:.4f} m/s"
        )
    if stats.max_ang_speed > 0.05:
        raise AssertionError(
            f"Zero-command angular drift too high: {stats.max_ang_speed:.4f} rad/s"
        )


def _check_single_thruster_signs() -> None:
    env = _make_env(decimation=4)
    try:
        stats = _rollout(
            env,
            total_time_s=1.5,
            pulse_thruster_idx=0,
            pulse_command=30.0,
            pulse_duration_s=0.5,
        )
    finally:
        env.close()

    if stats.max_surge_vel <= 0.02:
        raise AssertionError(
            "Single-thruster pulse did not create positive surge velocity. "
            f"Observed max surge: {stats.max_surge_vel:.4f} m/s"
        )
    if stats.min_yaw_vel >= -0.01:
        raise AssertionError(
            "Single-thruster pulse did not create expected negative yaw rate. "
            f"Observed min yaw: {stats.min_yaw_vel:.4f} rad/s"
        )


def _relative_diff(a: float, b: float) -> float:
    denom = max(abs(a), abs(b), 5e-3)
    return abs(a - b) / denom


def _check_decimation_consistency() -> None:
    env_dec1 = _make_env(decimation=1)
    try:
        stats_dec1 = _rollout(
            env_dec1,
            total_time_s=1.0,
            pulse_thruster_idx=0,
            pulse_command=25.0,
            pulse_duration_s=1.0,
        )
    finally:
        env_dec1.close()

    env_dec4 = _make_env(decimation=4)
    try:
        stats_dec4 = _rollout(
            env_dec4,
            total_time_s=1.0,
            pulse_thruster_idx=0,
            pulse_command=25.0,
            pulse_duration_s=1.0,
        )
    finally:
        env_dec4.close()

    surge_rel = _relative_diff(stats_dec1.final_surge_vel, stats_dec4.final_surge_vel)
    yaw_rel = _relative_diff(stats_dec1.final_yaw_vel, stats_dec4.final_yaw_vel)

    if surge_rel > 0.20:
        raise AssertionError(
            "Decimation consistency check failed for surge response: "
            f"rel_diff={surge_rel:.3f}, dec1={stats_dec1.final_surge_vel:.4f}, "
            f"dec4={stats_dec4.final_surge_vel:.4f}"
        )
    if yaw_rel > 0.20:
        raise AssertionError(
            "Decimation consistency check failed for yaw response: "
            f"rel_diff={yaw_rel:.3f}, dec1={stats_dec1.final_yaw_vel:.4f}, "
            f"dec4={stats_dec4.final_yaw_vel:.4f}"
        )


def main() -> None:
    _check_zero_action_trim()
    _check_single_thruster_signs()
    _check_decimation_consistency()
    print("Taluy MJLab thruster env smoke checks passed.")


if __name__ == "__main__":
    main()
