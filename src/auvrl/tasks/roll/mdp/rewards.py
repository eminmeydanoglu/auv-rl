from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import wrap_to_pi

from auvrl.tasks.roll.runtime import (
    action_term_slice,
    current_root_pose_from_qpos,
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def roll_progress(
    env: ManagerBasedRlEnv,
    roll_direction: int,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Reward signed positive progress along the commanded roll direction."""
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    state = get_roll_task_state(env, entity_name=entity_name)
    signed_delta = float(roll_direction) * state.delta_roll_rad
    return torch.clamp(signed_delta, min=0.0)


def xy_drift_penalty(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize drift in the world XY plane."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    xy_error_w = root_pos_w[:, :2] - state.xy_ref_w
    return -torch.linalg.vector_norm(xy_error_w, dim=1)


def pitch_penalty(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize absolute pitch magnitude."""
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    _roll_rad, pitch_rad, _yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
    return -pitch_rad.abs()


def yaw_hold_penalty(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize yaw drift from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    _roll_rad, _pitch_rad, yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
    yaw_error_rad = wrap_to_pi(yaw_rad - state.psi_ref_rad)
    return -yaw_error_rad.abs()


def depth_hold_penalty(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize depth error from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    depth_error_m = root_pos_w[:, 2] - state.z_ref_m
    return -depth_error_m.abs()


def body_wrench_action_rate_l2(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return ``||a_t - a_{t-1}||_2^2`` for the selected action term."""
    action_slice = action_term_slice(env, action_name)
    delta_action = (
        env.action_manager.action[:, action_slice]
        - env.action_manager.prev_action[:, action_slice]
    )
    return torch.sum(torch.square(delta_action), dim=1)


def terminal_success_reward(
    env: ManagerBasedRlEnv,
    termination_name: str = "task_success",
) -> torch.Tensor:
    """Emit 1.0 on success so config weights can set the terminal bonus."""
    return env.termination_manager.get_term(termination_name).float()


def terminal_failure_reward(
    env: ManagerBasedRlEnv,
    success_term_name: str = "task_success",
) -> torch.Tensor:
    """Emit 1.0 for terminal failures and timeouts, excluding task success."""
    success = env.termination_manager.get_term(success_term_name)
    return (env.reset_buf & ~success).float()


__all__ = [
    "body_wrench_action_rate_l2",
    "depth_hold_penalty",
    "pitch_penalty",
    "roll_progress",
    "terminal_failure_reward",
    "terminal_success_reward",
    "xy_drift_penalty",
    "yaw_hold_penalty",
]
