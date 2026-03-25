"""Domain-randomization event terms for MJLab AUV environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from auvrl.actuator.thruster_actuator import ThrusterActuator
from auvrl.sim.underwater_hydro_action import UnderwaterHydroAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


def _selected_env_ids(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(
            env.num_envs,
            device=env.device,
            dtype=torch.long,
        )[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def _sample_uniform(
    low: float,
    high: float,
    shape: tuple[int, ...],
    *,
    device: str,
) -> torch.Tensor:
    if high < low:
        raise ValueError(f"Expected low <= high, got ({low}, {high}).")
    if high == low:
        return torch.full(shape, low, device=device, dtype=torch.float)
    return torch.rand(shape, device=device, dtype=torch.float) * (high - low) + low


def randomize_thruster_supply_voltage(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    entity_name: str = "robot",
    voltage_range: tuple[float, float] = (14.0, 18.0),
) -> None:
    """Randomize thruster supply voltage per environment."""
    selected_env_ids = _selected_env_ids(env, env_ids)
    if selected_env_ids.numel() == 0:
        return

    entity = env.scene[entity_name]
    thruster_actuators = [
        actuator
        for actuator in entity.actuators
        if isinstance(actuator, ThrusterActuator)
    ]
    if len(thruster_actuators) != 1:
        raise ValueError(
            f"Expected exactly one ThrusterActuator on '{entity_name}', got "
            f"{len(thruster_actuators)}."
        )
    thruster_actuator = thruster_actuators[0]
    num_selected_envs = int(selected_env_ids.numel())
    num_thrusters = len(thruster_actuator.target_names)

    sampled_voltage = _sample_uniform(
        voltage_range[0],
        voltage_range[1],
        (num_selected_envs, 1),
        device=env.device,
    )
    thruster_actuator.set_supply_voltage(
        sampled_voltage.expand(num_selected_envs, num_thrusters).clone(),
        env_ids=selected_env_ids,
    )


def randomize_water_current_velocity(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    action_term_name: str = "hydro",
    speed_range_m_s: tuple[float, float] = (0.0, 0.5),
    yaw_range_rad: tuple[float, float] = (
        -3.141592653589793,
        3.141592653589793,
    ),
    vertical_range_m_s: tuple[float, float] = (-0.1, 0.1),
) -> None:
    """Randomize world-frame water current from speed/yaw/vertical ranges."""
    selected_env_ids = _selected_env_ids(env, env_ids)
    if selected_env_ids.numel() == 0:
        return

    action_term = env.action_manager.get_term(action_term_name)
    if not isinstance(action_term, UnderwaterHydroAction):
        raise ValueError(
            f"Action term '{action_term_name}' is not UnderwaterHydroAction."
        )
    num_selected_envs = int(selected_env_ids.numel())

    speed = _sample_uniform(
        speed_range_m_s[0],
        speed_range_m_s[1],
        (num_selected_envs, 1),
        device=env.device,
    )
    yaw = _sample_uniform(
        yaw_range_rad[0],
        yaw_range_rad[1],
        (num_selected_envs, 1),
        device=env.device,
    )
    vertical = _sample_uniform(
        vertical_range_m_s[0],
        vertical_range_m_s[1],
        (num_selected_envs, 1),
        device=env.device,
    )

    current_velocity_w = torch.cat(
        (speed * torch.cos(yaw), speed * torch.sin(yaw), vertical),
        dim=1,
    )
    action_term.set_current_velocity_w(current_velocity_w, env_ids=selected_env_ids)
