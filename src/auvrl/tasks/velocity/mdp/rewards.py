from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from auvrl.actuator.body_wrench_action import BodyWrenchAction

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def track_body_linear_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward 3D body-frame linear-velocity tracking with a Gaussian kernel."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    robot: Entity = env.scene[entity_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    error = command[:, :3] - robot.data.root_link_lin_vel_b
    return torch.exp(-torch.sum(torch.square(error), dim=1) / float(std * std))


def track_body_angular_velocity(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward 3D body-frame angular-velocity tracking with a Gaussian kernel."""
    if std <= 0.0:
        raise ValueError(f"std must be positive, got {std}.")

    robot: Entity = env.scene[entity_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."

    error = command[:, 3:] - robot.data.root_link_ang_vel_b
    return torch.exp(-torch.sum(torch.square(error), dim=1) / float(std * std))


def body_wrench_action_l2(
    env: ManagerBasedRlEnv,
    action_name: str,
    wrench_limits: tuple[float, float, float, float, float, float] | None = None,
) -> torch.Tensor:
    """Penalize commanded body wrench after normalizing by axis limits."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")

    if wrench_limits is None:
        return torch.mean(torch.square(action_term.raw_action), dim=1)

    limits = torch.tensor(wrench_limits, dtype=torch.float, device=env.device).view(
        1, 6
    )
    if torch.any(limits <= 0.0):
        raise ValueError("wrench_limits must all be positive.")

    normalized_action = action_term.desired_wrench_b / limits
    return torch.mean(torch.square(normalized_action), dim=1)


def body_wrench_action_rate_l2(
    env: ManagerBasedRlEnv,
    wrench_limits: tuple[float, float, float, float, float, float] | None = None,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Penalize normalized rate-of-change in body-wrench policy outputs."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")

    if wrench_limits is None:
        delta = env.action_manager.action - env.action_manager.prev_action
        return torch.mean(torch.square(delta), dim=1)

    limits = torch.tensor(wrench_limits, dtype=torch.float, device=env.device).view(
        1, 6
    )
    if torch.any(limits <= 0.0):
        raise ValueError("wrench_limits must all be positive.")

    current_wrench = action_term.action_to_wrench(env.action_manager.action)
    prev_wrench = action_term.action_to_wrench(env.action_manager.prev_action)
    delta = (current_wrench - prev_wrench) / limits
    return torch.mean(torch.square(delta), dim=1)


def body_wrench_saturation_penalty(
    env: ManagerBasedRlEnv,
    action_name: str,
) -> torch.Tensor:
    """Penalize action-allocation saturation on the thruster limits."""
    action_term = env.action_manager.get_term(action_name)
    if not isinstance(action_term, BodyWrenchAction):
        raise ValueError(f"Action term '{action_name}' is not BodyWrenchAction.")
    return action_term.step_saturation_fraction
