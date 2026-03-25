"""Smoke checks for the project-local MJLab thruster actuator."""

from __future__ import annotations

import math
from pathlib import Path

try:
    import mujoco
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'mujoco'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.entity import Entity, EntityArticulationInfoCfg, EntityCfg
    from mjlab.sim.sim import Simulation, SimulationCfg
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. Ensure mjlab is available (local ../mjlab or installed dependency)."
    ) from exc

from auvrl import ThrusterActuatorCfg  # type: ignore[import-not-found]  # noqa: E402

XML = """
<mujoco>
  <option timestep="0.01"/>
  <worldbody>
    <body name="base" pos="0 0 0">
      <freejoint name="root"/>
      <geom name="hull" type="box" size="0.2 0.1 0.1" mass="8.0"/>
      <site name="thruster_left" pos="0.0 0.1 0.0"/>
      <site name="thruster_right" pos="0.0 -0.1 0.0"/>
    </body>
  </worldbody>
</mujoco>
"""


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_entity(cfg: ThrusterActuatorCfg) -> Entity:
    return Entity(
        EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_string(XML),
            articulation=EntityArticulationInfoCfg(actuators=(cfg,)),
        )
    )


def _initialize(entity: Entity, device: str) -> tuple[Entity, Simulation]:
    model = entity.compile()
    sim = Simulation(num_envs=1, cfg=SimulationCfg(), model=model, device=device)
    entity.initialize(model, sim.model, sim.data, device)
    return entity, sim


def _check_force_passthrough(device: str) -> None:
    cfg = ThrusterActuatorCfg(
        target_names_expr=("thruster_.*",),
        tau_s=1e-6,
        command_limit=100.0,
        force_deadzone_n=0.0,
        min_thrust_n=-40.0,
        max_thrust_n=60.0,
        supply_voltage=16.0,
        pwm_min_us=1100.0,
        pwm_max_us=1900.0,
        pwm_neutral_us=1500.0,
        force_to_pwm_coeffs_forward=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        force_to_pwm_coeffs_reverse=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        newton_per_kgf=10.0,
    )
    entity, sim = _initialize(_make_entity(cfg), device)
    entity.set_site_effort_target(torch.tensor([[30.0, -20.0]], device=device))
    entity.write_data_to_sim()
    ctrl = sim.data.ctrl[0]
    expected = torch.tensor([30.0, -20.0], device=device)
    torch.testing.assert_close(ctrl, expected, atol=1e-6, rtol=1e-6)


def _check_first_order_lag(device: str) -> None:
    cfg = ThrusterActuatorCfg(
        target_names_expr=("thruster_.*",),
        tau_s=0.02,
        command_limit=20.0,
        force_deadzone_n=0.0,
        min_thrust_n=-20.0,
        max_thrust_n=20.0,
        supply_voltage=16.0,
        pwm_min_us=1100.0,
        pwm_max_us=1900.0,
        pwm_neutral_us=1500.0,
        force_to_pwm_coeffs_forward=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        force_to_pwm_coeffs_reverse=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        newton_per_kgf=10.0,
    )
    entity, sim = _initialize(_make_entity(cfg), device)

    command = torch.tensor([[10.0, 10.0]], device=device)
    entity.set_site_effort_target(command)
    entity.write_data_to_sim()
    first = sim.data.ctrl[0].clone()

    entity.set_site_effort_target(command)
    entity.write_data_to_sim()
    second = sim.data.ctrl[0].clone()

    dt_s = float(sim.mj_model.opt.timestep)
    alpha = math.exp(-dt_s / cfg.tau_s)
    expected_first = (1.0 - alpha) * 10.0
    expected_second = alpha * expected_first + (1.0 - alpha) * 10.0

    expected_first_tensor = torch.tensor(
        [expected_first, expected_first], device=device
    )
    expected_second_tensor = torch.tensor(
        [expected_second, expected_second],
        device=device,
    )
    torch.testing.assert_close(first, expected_first_tensor, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(second, expected_second_tensor, atol=1e-6, rtol=1e-6)


def _check_force_target_saturation(device: str) -> None:
    cfg = ThrusterActuatorCfg(
        target_names_expr=("thruster_.*",),
        tau_s=1e-6,
        command_limit=500.0,
        force_deadzone_n=0.0,
        min_thrust_n=-100.0,
        max_thrust_n=100.0,
        supply_voltage=16.0,
        pwm_min_us=1490.0,
        pwm_max_us=1510.0,
        pwm_neutral_us=1500.0,
        force_to_pwm_coeffs_forward=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        force_to_pwm_coeffs_reverse=(0.0, 0.0, 0.0, 1.0, 0.0, 1500.0),
        newton_per_kgf=10.0,
    )
    entity, sim = _initialize(_make_entity(cfg), device)

    entity.set_site_effort_target(torch.tensor([[500.0, -500.0]], device=device))
    entity.write_data_to_sim()
    ctrl = sim.data.ctrl[0]
    expected = torch.tensor([100.0, -100.0], device=device)
    torch.testing.assert_close(ctrl, expected, atol=1e-6, rtol=1e-6)


def main() -> None:
    device = _device()
    _check_force_passthrough(device)
    _check_first_order_lag(device)
    _check_force_target_saturation(device)
    print("MJLab thruster actuator smoke checks passed.")


if __name__ == "__main__":
    main()
