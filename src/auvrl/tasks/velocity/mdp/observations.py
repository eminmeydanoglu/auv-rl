from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from auvrl.actuator.body_wrench_action import BodyWrenchAction
from auvrl.actuator.thruster_actuator import ThrusterActuator
from auvrl.sim.underwater_hydro_action import UnderwaterHydroAction

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def thruster_force_state(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return normalized per-thruster thrust after actuator lag/saturation."""
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
    limit = max(float(thruster_actuator.cfg.command_limit), 1.0e-6)
    return thruster_actuator.thrust_state / limit


def current_velocity_b(
    env: ManagerBasedRlEnv,
    action_name: str = "hydro",
) -> torch.Tensor:
    """Return water current expressed in the vehicle body frame."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, UnderwaterHydroAction):
        raise ValueError(f"Action term '{action_name}' is not UnderwaterHydroAction.")
    return action_term.current_velocity_b


def depth_error(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return root depth offset from the environment origin."""
    robot: Entity = env.scene[entity_name]
    return (robot.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]).unsqueeze(1)


def thruster_voltage_offset(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    nominal_voltage_v: float = 16.0,
    scale_v: float = 2.0,
) -> torch.Tensor:
    """Return normalized supply-voltage offset averaged over thrusters."""
    if scale_v <= 0.0:
        raise ValueError(f"scale_v must be positive, got {scale_v}.")
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
    mean_voltage = thruster_actuator.supply_voltage.mean(dim=1, keepdim=True)
    return (mean_voltage - float(nominal_voltage_v)) / float(scale_v)


def applied_body_wrench(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
    normalize: bool = True,
) -> torch.Tensor:
    """Return the last allocated body wrench, optionally normalized by limits."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")
    wrench = action_term.applied_wrench_origin_b
    if not normalize:
        return wrench
    return wrench / action_term.wrench_limit
