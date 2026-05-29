from __future__ import annotations

import math
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
    target_roll_rad: float,
    progress_normalization_rad: float | None = None,
    entity_name: str = "robot",
) -> torch.Tensor:
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    if target_roll_rad <= 0.0:
        raise ValueError(f"target_roll_rad must be positive, got {target_roll_rad}.")
    if progress_normalization_rad is None:
        progress_normalization_rad = min(float(target_roll_rad), math.pi)
    if progress_normalization_rad <= 0.0:
        raise ValueError(
            "progress_normalization_rad must be positive, "
            f"got {progress_normalization_rad}."
        )

    state = get_roll_task_state(env, entity_name=entity_name)

    # After target roll achieved, dont reward roll progress
    signed_phi_after = float(roll_direction) * state.phi_total_rad
    signed_phi_before = signed_phi_after - float(roll_direction) * state.delta_roll_rad

    target = torch.as_tensor(
        float(target_roll_rad),
        dtype=signed_phi_after.dtype,
        device=signed_phi_after.device,
    )
    bounded_before = torch.minimum(signed_phi_before, target)
    bounded_after = torch.minimum(signed_phi_after, target)
    progress_norm = torch.as_tensor(
        float(progress_normalization_rad),
        dtype=signed_phi_after.dtype,
        device=signed_phi_after.device,
    )
    bounded_progress_delta = (bounded_after - bounded_before) / progress_norm
    return torch.clamp(bounded_progress_delta, min=0.0)


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


def nonroll_body_wrench_action_rate_l2(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
    nonroll_indices: tuple[int, ...] = (0, 1, 2, 4, 5),
) -> torch.Tensor:
    """Return action-rate L2 on body-wrench axes except roll torque."""
    action_slice = action_term_slice(env, action_name)
    action = env.action_manager.action[:, action_slice]
    if not nonroll_indices:
        return torch.zeros(env.num_envs, dtype=action.dtype, device=action.device)
    invalid_indices = [
        index for index in nonroll_indices if not 0 <= int(index) < action.shape[1]
    ]
    if invalid_indices:
        raise ValueError(
            "nonroll_indices must be valid body-wrench action indices, "
            f"got {tuple(invalid_indices)} for action dim {action.shape[1]}."
        )
    delta_action = (
        action[:, list(nonroll_indices)]
        - env.action_manager.prev_action[:, action_slice][:, list(nonroll_indices)]
    )
    return torch.sum(torch.square(delta_action), dim=1)


def body_wrench_action_effort(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return mean squared normalized body-wrench action as an effort cost."""
    action_slice = action_term_slice(env, action_name)
    action = env.action_manager.action[:, action_slice]
    return torch.mean(torch.square(action), dim=1)


def post_target_roll_through_torque_penalty(
    env: ManagerBasedRlEnv,
    roll_direction: int,
    target_roll_rad: float,
    action_name: str = "body_wrench",
    tx_index: int = 3,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Penalize same-direction roll torque after the roll target is reached."""
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    if target_roll_rad <= 0.0:
        raise ValueError(f"target_roll_rad must be positive, got {target_roll_rad}.")

    action_slice = action_term_slice(env, action_name)
    action = env.action_manager.action[:, action_slice]
    if not 0 <= tx_index < action.shape[1]:
        raise ValueError(
            f"tx_index must be in [0, {action.shape[1]}), got {tx_index}."
        )

    state = get_roll_task_state(env, entity_name=entity_name)
    signed_progress = float(roll_direction) * state.phi_total_rad
    post_target = state.target_reached | (signed_progress >= float(target_roll_rad))
    roll_through_torque = torch.clamp(
        float(roll_direction) * action[:, tx_index],
        min=0.0,
    )
    return post_target.float() * torch.square(roll_through_torque)


def thruster_saturation_cost(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
    threshold: float = 0.85,
    max_thruster_n: float | None = None,
) -> torch.Tensor:
    """Return a soft cost for thruster targets above a normalized threshold."""
    if not 0.0 <= threshold < 1.0:
        raise ValueError(f"threshold must be in [0, 1), got {threshold}.")

    term = env.action_manager.get_term(action_name)
    thruster_targets = getattr(term, "thruster_targets", None)
    if thruster_targets is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    if max_thruster_n is None:
        max_thruster_n = getattr(
            getattr(term, "cfg", None),
            "site_force_limit_n",
            None,
        )
    if max_thruster_n is None:
        max_thruster_n = torch.max(thruster_targets.abs()).detach().clamp_min(1.0e-6)
    limit = torch.as_tensor(
        max_thruster_n,
        dtype=thruster_targets.dtype,
        device=thruster_targets.device,
    ).clamp_min(1.0e-6)
    normalized = thruster_targets.abs() / limit
    excess = torch.clamp(normalized - float(threshold), min=0.0)
    return torch.mean(torch.square(excess), dim=1)


def terminal_success_reward(
    env: ManagerBasedRlEnv,
    termination_name: str = "task_success",
) -> torch.Tensor:
    """Emit 1.0 on success so config weights can set the terminal bonus."""
    try:
        success = env.termination_manager.get_term(termination_name)
    except KeyError:
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return success.float()


def terminal_failure_reward(
    env: ManagerBasedRlEnv,
    success_term_name: str = "task_success",
) -> torch.Tensor:
    """Emit 1.0 for terminal failures and timeouts, excluding task success."""
    try:
        success = env.termination_manager.get_term(success_term_name)
    except KeyError:
        success = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    return (env.reset_buf & ~success).float()


__all__ = [
    "body_wrench_action_effort",
    "body_wrench_action_rate_l2",
    "depth_hold_penalty",
    "nonroll_body_wrench_action_rate_l2",
    "pitch_penalty",
    "post_target_roll_through_torque_penalty",
    "roll_progress",
    "terminal_failure_reward",
    "terminal_success_reward",
    "thruster_saturation_cost",
    "xy_drift_penalty",
    "yaw_hold_penalty",
]
