from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import wrap_to_pi

from auvrl.tasks.roll.runtime import (
    action_term_slice,
    current_root_ang_vel_b_from_qvel,
    current_root_pose_from_qpos,
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def roll_progress_ratio(
    env: ManagerBasedRlEnv,
    *,
    target_roll_rad: float,
    roll_direction: int,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return signed cumulative roll progress divided by the task target."""
    if target_roll_rad <= 0.0:
        raise ValueError(f"target_roll_rad must be positive, got {target_roll_rad}.")
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    state = get_roll_task_state(env, entity_name=entity_name)
    return float(roll_direction) * state.phi_total_rad / float(target_roll_rad)


def phi_total_rad(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return cumulative unwrapped roll in radians."""
    return get_roll_task_state(env, entity_name=entity_name).phi_total_rad


def target_reached(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return whether the cumulative roll target has been reached."""
    return get_roll_task_state(env, entity_name=entity_name).target_reached.float()


def settle_counter_s(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return the current continuous settle-window counter in seconds."""
    state = get_roll_task_state(env, entity_name=entity_name)
    return state.settle_counter_steps.float() * float(env.step_dt)


def depth_abs_error_m(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return absolute depth error from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    return (root_pos_w[:, 2] - state.z_ref_m).abs()


def xy_drift_m(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return world-frame XY drift magnitude from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    return torch.linalg.vector_norm(root_pos_w[:, :2] - state.xy_ref_w, dim=1)


def pitch_abs_rad(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return absolute pitch angle in radians."""
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    _roll_rad, pitch_rad, _yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
    return pitch_rad.abs()


def yaw_abs_error_rad(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return absolute yaw error from the reset-time reference in radians."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    _roll_rad, _pitch_rad, yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
    return wrap_to_pi(yaw_rad - state.psi_ref_rad).abs()


def root_ang_speed_rad_s(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return root angular speed magnitude in body frame."""
    robot: Entity = env.scene[entity_name]
    return torch.linalg.vector_norm(current_root_ang_vel_b_from_qvel(robot), dim=1)


def body_wrench_action_l2(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return squared L2 norm of the current normalized body-wrench action."""
    action_slice = action_term_slice(env, action_name)
    action = env.action_manager.action[:, action_slice]
    return torch.sum(torch.square(action), dim=1)


def body_wrench_saturation_fraction(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return fraction of allocated thrusters saturated this step."""
    term = env.action_manager.get_term(action_name)
    saturation = getattr(term, "step_saturation_fraction", None)
    if saturation is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    return saturation.float()


def water_current_speed_m_s(
    env: ManagerBasedRlEnv,
    action_name: str = "hydro",
) -> torch.Tensor:
    """Return world-frame water-current speed magnitude."""
    term = env.action_manager.get_term(action_name)
    current_velocity_w = getattr(term, "current_velocity_w", None)
    if current_velocity_w is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    return torch.linalg.vector_norm(current_velocity_w, dim=1)


def hydro_wrench_norm(
    env: ManagerBasedRlEnv,
    action_name: str = "hydro",
) -> torch.Tensor:
    """Return norm of the latest hydrodynamic body-frame wrench."""
    term = env.action_manager.get_term(action_name)
    applied_wrench_b = getattr(term, "applied_wrench_b", None)
    if applied_wrench_b is None:
        return torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    return torch.linalg.vector_norm(applied_wrench_b, dim=1)


__all__ = [
    "body_wrench_action_l2",
    "body_wrench_saturation_fraction",
    "depth_abs_error_m",
    "hydro_wrench_norm",
    "phi_total_rad",
    "pitch_abs_rad",
    "roll_progress_ratio",
    "root_ang_speed_rad_s",
    "settle_counter_s",
    "target_reached",
    "water_current_speed_m_s",
    "xy_drift_m",
    "yaw_abs_error_rad",
]
