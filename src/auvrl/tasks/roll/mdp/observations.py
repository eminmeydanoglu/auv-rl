from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from auvrl.tasks.roll.runtime import (
    action_term_slice,
    current_root_pose_from_qpos,
    get_roll_task_state,
    normalized_phi_total,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def base_quat_wxyz(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return the root-link orientation quaternion in world frame."""
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    return quat_wxyz


def depth_error_from_ref(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return depth error relative to the reset-time depth reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    depth_error = root_pos_w[:, 2] - state.z_ref_m
    return depth_error.unsqueeze(1)


def phi_total_norm(
    env: ManagerBasedRlEnv,
    *,
    target_roll_rad: float,
    roll_direction: int,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return the clipped normalized cumulative roll progress."""
    state = get_roll_task_state(env, entity_name=entity_name)
    normalized = normalized_phi_total(
        state.phi_total_rad,
        target_roll_rad=target_roll_rad,
        roll_direction=roll_direction,
    )
    return normalized.unsqueeze(1)


def xy_error_w(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return world-frame XY drift from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    return root_pos_w[:, :2] - state.xy_ref_w


def last_body_wrench_action(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return the previous body-wrench policy action ``a_{t-1}``."""
    action_slice = action_term_slice(env, action_name)
    return env.action_manager.prev_action[:, action_slice]


__all__ = [
    "base_quat_wxyz",
    "depth_error_from_ref",
    "last_body_wrench_action",
    "phi_total_norm",
    "xy_error_w",
]
