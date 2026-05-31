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
    return torch.linalg.vector_norm(_xy_error_w(env, entity_name=entity_name), dim=1)


def _xy_error_w(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return signed world-frame XY error from the reset-time reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    return root_pos_w[:, :2] - state.xy_ref_w


def x_drift_m(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return signed world-frame X drift from the reset-time reference."""
    return _xy_error_w(env, entity_name=entity_name)[:, 0]


def y_drift_m(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return signed world-frame Y drift from the reset-time reference."""
    return _xy_error_w(env, entity_name=entity_name)[:, 1]


class XyDriftPeakM:
    """Track the peak XY drift reached within each episode."""

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._peak = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        current = xy_drift_m(env, entity_name=entity_name)
        self._peak = torch.maximum(self._peak, current)
        return self._peak

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._peak.zero_()
        else:
            self._peak[env_ids] = 0.0


class PitchAbsPeakRad:
    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._peak = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        entity_name: str = "robot",
    ) -> torch.Tensor:
        current = pitch_abs_rad(env, entity_name=entity_name)
        self._peak = torch.maximum(self._peak, current)
        return self._peak

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._peak.zero_()
        else:
            self._peak[env_ids] = 0.0


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


def body_wrench_action_rate_l2(
    env: ManagerBasedRlEnv,
    action_name: str = "body_wrench",
) -> torch.Tensor:
    """Return squared L2 norm of the normalized body-wrench action delta."""
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
    delta_action = (
        action[:, list(nonroll_indices)]
        - env.action_manager.prev_action[:, action_slice][:, list(nonroll_indices)]
    )
    return torch.sum(torch.square(delta_action), dim=1)


def post_target_roll_through_torque(
    env: ManagerBasedRlEnv,
    roll_direction: int,
    target_roll_rad: float,
    action_name: str = "body_wrench",
    tx_index: int = 3,
    entity_name: str = "robot",
) -> torch.Tensor:
    """Return same-direction roll torque squared after the roll target."""
    action_slice = action_term_slice(env, action_name)
    action = env.action_manager.action[:, action_slice]
    state = get_roll_task_state(env, entity_name=entity_name)
    signed_progress = float(roll_direction) * state.phi_total_rad
    post_target = state.target_reached | (signed_progress >= float(target_roll_rad))
    roll_through_torque = torch.clamp(
        float(roll_direction) * action[:, tx_index],
        min=0.0,
    )
    return post_target.float() * torch.square(roll_through_torque)


class PostTargetRollThroughTorqueMean:
    """Track mean same-direction roll torque squared after the roll target."""

    def __init__(self, cfg: object, env: ManagerBasedRlEnv) -> None:
        del cfg
        self._sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
        self._count = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        roll_direction: int,
        target_roll_rad: float,
        action_name: str = "body_wrench",
        tx_index: int = 3,
        entity_name: str = "robot",
    ) -> torch.Tensor:
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
        post_target_f = post_target.float()
        self._sum += post_target_f * torch.square(roll_through_torque)
        self._count += post_target_f
        return torch.where(
            self._count > 0.0,
            self._sum / self._count.clamp_min(1.0),
            torch.zeros_like(self._sum),
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            self._sum.zero_()
            self._count.zero_()
        else:
            self._sum[env_ids] = 0.0
            self._count[env_ids] = 0.0


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
    "body_wrench_action_rate_l2",
    "body_wrench_saturation_fraction",
    "depth_abs_error_m",
    "hydro_wrench_norm",
    "nonroll_body_wrench_action_rate_l2",
    "phi_total_rad",
    "PitchAbsPeakRad",
    "pitch_abs_rad",
    "PostTargetRollThroughTorqueMean",
    "post_target_roll_through_torque",
    "roll_progress_ratio",
    "root_ang_speed_rad_s",
    "settle_counter_s",
    "target_reached",
    "water_current_speed_m_s",
    "x_drift_m",
    "XyDriftPeakM",
    "xy_drift_m",
    "y_drift_m",
    "yaw_abs_error_rad",
]
