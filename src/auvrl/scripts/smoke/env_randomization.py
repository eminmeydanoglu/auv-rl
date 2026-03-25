"""Smoke checks for Taluy MJLab env randomization events."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. "
        "Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. "
        "Ensure mjlab is available (local ../mjlab or installed dependency)."
    ) from exc

try:
    from auvrl.scripts.smoke import env as env_smoke
except ModuleNotFoundError as exc:
    raise SystemExit("Could not import the base env smoke module.") from exc

from auvrl import make_taluy_base_env_cfg  # type: ignore[import-not-found]  # noqa: E402
from auvrl.actuator.thruster_actuator import ThrusterActuator  # type: ignore[import-not-found]  # noqa: E402
from auvrl.sim.underwater_hydro_action import (  # type: ignore[import-not-found]  # noqa: E402
    UnderwaterHydroAction,
)


THRUSTER_VOLTAGE_RANGE_V = (14.0, 18.0)
CURRENT_SPEED_RANGE_M_S = (0.1, 0.6)
CURRENT_YAW_RANGE_RAD = (-1.2, 1.2)
VERTICAL_CURRENT_RANGE_M_S = (-0.08, 0.08)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_in_range(
    name: str,
    values: torch.Tensor,
    low: float,
    high: float,
) -> None:
    eps = 1e-6
    min_value = float(values.min().item())
    max_value = float(values.max().item())
    if min_value < low - eps or max_value > high + eps:
        raise AssertionError(
            f"{name} out of range [{low}, {high}]: min={min_value}, max={max_value}"
        )


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _make_randomized_env() -> ManagerBasedRlEnv:
    cfg = make_taluy_base_env_cfg(
        action_space="thruster",
        thruster_voltage_event_mode="reset",
        thruster_voltage_range_v=THRUSTER_VOLTAGE_RANGE_V,
        current_event_mode="reset",
        current_speed_range_m_s=CURRENT_SPEED_RANGE_M_S,
        current_yaw_range_rad=CURRENT_YAW_RANGE_RAD,
        vertical_current_range_m_s=VERTICAL_CURRENT_RANGE_M_S,
    )
    cfg.scene.num_envs = 8
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _check_randomized_reset_values() -> None:
    env = _make_randomized_env()
    try:
        action = torch.zeros(
            (env.num_envs, env.action_manager.total_action_dim),
            device=env.device,
        )

        for _ in range(32):
            env.reset()

            robot = env.scene["robot"]
            thruster_actuators = [
                actuator
                for actuator in robot.actuators
                if isinstance(actuator, ThrusterActuator)
            ]
            if len(thruster_actuators) != 1:
                raise AssertionError(
                    "Expected exactly one ThrusterActuator on 'robot', got "
                    f"{len(thruster_actuators)}"
                )
            thruster = thruster_actuators[0]

            hydro = env.action_manager.get_term("hydro")
            if not isinstance(hydro, UnderwaterHydroAction):
                raise AssertionError("'hydro' action term is not UnderwaterHydroAction")

            voltage = thruster.supply_voltage
            _assert_in_range(
                "thruster_voltage",
                voltage,
                THRUSTER_VOLTAGE_RANGE_V[0],
                THRUSTER_VOLTAGE_RANGE_V[1],
            )

            current = hydro.current_velocity_w
            speed_xy = torch.linalg.norm(current[:, :2], dim=1)
            yaw = torch.atan2(current[:, 1], current[:, 0])
            vertical = current[:, 2]

            _assert_in_range(
                "current_speed_xy",
                speed_xy,
                CURRENT_SPEED_RANGE_M_S[0],
                CURRENT_SPEED_RANGE_M_S[1],
            )
            _assert_in_range(
                "current_yaw",
                yaw,
                CURRENT_YAW_RANGE_RAD[0],
                CURRENT_YAW_RANGE_RAD[1],
            )
            _assert_in_range(
                "current_vertical",
                vertical,
                VERTICAL_CURRENT_RANGE_M_S[0],
                VERTICAL_CURRENT_RANGE_M_S[1],
            )

            for _ in range(6):
                env.step(action)
                _assert_finite_env_state(env)
    finally:
        env.close()


def _check_decimation_baseline_when_randomization_disabled() -> None:
    env_smoke._check_decimation_consistency()


def main() -> None:
    _check_randomized_reset_values()
    _check_decimation_baseline_when_randomization_disabled()
    print("MJLab env randomization smoke checks passed.")


if __name__ == "__main__":
    main()
