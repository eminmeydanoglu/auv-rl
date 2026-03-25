"""Headless regression check for Taluy MJLab dynamics wiring."""

from __future__ import annotations

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
        "Could not import mjlab. Ensure mjlab is available "
        "(local ../mjlab or installed dependency)."
    ) from exc

from auvrl import make_taluy_base_env_cfg  # type: ignore[import-not-found]  # noqa: E402
from auvrl.config.auv_cfg import (  # type: ignore[import-not-found]  # noqa: E402
    TALUY_CFG_PATH,
    load_auv_cfg,
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _tail_mean(samples: torch.Tensor, fraction: float = 0.35) -> torch.Tensor:
    if samples.shape[0] == 0:
        raise AssertionError("Expected non-empty phase samples")
    count = max(int(round(samples.shape[0] * fraction)), 1)
    return samples[-count:].mean(dim=0)


def _run_phase(
    env: ManagerBasedRlEnv,
    *,
    command_8: torch.Tensor,
    duration_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if command_8.shape != (8,):
        raise ValueError(f"Expected command shape (8,), got {tuple(command_8.shape)}")

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    action[:, :8] = command_8.view(1, 8)

    num_steps = max(int(duration_s / env.step_dt), 1)
    lin_hist: list[torch.Tensor] = []
    ang_hist: list[torch.Tensor] = []

    for _ in range(num_steps):
        env.step(action)
        _assert_finite_env_state(env)

        robot = env.scene["robot"]
        lin_hist.append(robot.data.root_link_lin_vel_b[0].detach().cpu())
        ang_hist.append(robot.data.root_link_ang_vel_b[0].detach().cpu())

    return torch.stack(lin_hist), torch.stack(ang_hist)


def main() -> None:
    cfg = make_taluy_base_env_cfg(
        action_space="thruster",
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=_device())

    try:
        env.reset()

        taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)
        coast_cmd = torch.as_tensor(
            taluy_cfg.coast_command,
            device=env.device,
            dtype=torch.float,
        )
        surge_cmd = torch.as_tensor(
            taluy_cfg.surge_command,
            device=env.device,
            dtype=torch.float,
        )
        yaw_cmd = torch.as_tensor(
            taluy_cfg.yaw_command,
            device=env.device,
            dtype=torch.float,
        )
        heave_cmd = torch.as_tensor(
            taluy_cfg.heave_command,
            device=env.device,
            dtype=torch.float,
        )

        _run_phase(env, command_8=coast_cmd, duration_s=0.5)
        surge_lin, surge_ang = _run_phase(
            env,
            command_8=surge_cmd,
            duration_s=1.5,
        )
        yaw_lin, yaw_ang = _run_phase(
            env,
            command_8=yaw_cmd,
            duration_s=1.5,
        )
        heave_lin, heave_ang = _run_phase(
            env,
            command_8=heave_cmd,
            duration_s=1.5,
        )
        coast_lin, coast_ang = _run_phase(
            env,
            command_8=coast_cmd,
            duration_s=1.2,
        )

        surge_tail_lin = _tail_mean(surge_lin)
        yaw_tail_ang = _tail_mean(yaw_ang)
        heave_tail_lin = _tail_mean(heave_lin)

        if float(surge_tail_lin[0]) <= 0.04:
            raise AssertionError(
                "Surge phase did not produce positive body surge velocity. "
                f"Observed tail mean vx={float(surge_tail_lin[0]):.4f} m/s"
            )

        if abs(float(yaw_tail_ang[2])) <= 0.04:
            raise AssertionError(
                "Yaw phase did not produce yaw-rate response. "
                f"Observed tail mean wz={float(yaw_tail_ang[2]):.4f} rad/s"
            )

        if float(heave_tail_lin[2]) <= 0.03:
            raise AssertionError(
                "Heave phase did not produce positive body heave velocity. "
                f"Observed tail mean vz={float(heave_tail_lin[2]):.4f} m/s"
            )

        surge_peak = float(torch.max(torch.abs(surge_lin[:, 0])).item())
        heave_peak = float(torch.max(torch.abs(heave_lin[:, 2])).item())
        yaw_peak = float(torch.max(torch.abs(yaw_ang[:, 2])).item())

        coast_tail_lin = _tail_mean(coast_lin)
        coast_tail_ang = _tail_mean(coast_ang)
        coast_tail_lin_speed = float(torch.linalg.norm(coast_tail_lin).item())
        coast_tail_ang_speed = float(torch.linalg.norm(coast_tail_ang).item())

        lin_ref = max(surge_peak, heave_peak, 1.0e-6)
        ang_ref = max(yaw_peak, 1.0e-6)

        if coast_tail_lin_speed >= 0.95 * lin_ref:
            raise AssertionError(
                "Coast phase did not reduce linear speed as expected. "
                f"coast_tail_lin_speed={coast_tail_lin_speed:.4f}, "
                f"reference_peak={lin_ref:.4f}"
            )

        if coast_tail_ang_speed >= 0.95 * ang_ref:
            raise AssertionError(
                "Coast phase did not reduce angular speed as expected. "
                f"coast_tail_ang_speed={coast_tail_ang_speed:.4f}, "
                f"reference_peak={ang_ref:.4f}"
            )

    finally:
        env.close()

    print("Taluy MJLab dynamics regression smoke passed.")
    print(f"surge_tail_vx={float(surge_tail_lin[0]):.4f} m/s")
    print(f"yaw_tail_wz={float(yaw_tail_ang[2]):.4f} rad/s")
    print(f"heave_tail_vz={float(heave_tail_lin[2]):.4f} m/s")
    print(f"coast_tail_lin_speed={coast_tail_lin_speed:.4f} m/s")
    print(f"coast_tail_ang_speed={coast_tail_ang_speed:.4f} rad/s")
    print(
        "tail_lin_velocities: "
        f"surge={surge_tail_lin.tolist()} yaw={_tail_mean(yaw_lin).tolist()} "
        f"heave={heave_tail_lin.tolist()}"
    )
    print(
        "tail_ang_velocities: "
        f"surge={_tail_mean(surge_ang).tolist()} yaw={yaw_tail_ang.tolist()} "
        f"heave={_tail_mean(heave_ang).tolist()}"
    )


if __name__ == "__main__":
    main()
