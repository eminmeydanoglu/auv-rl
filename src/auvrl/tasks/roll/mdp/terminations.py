from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import wrap_to_pi

from auvrl.tasks.roll.runtime import (
    current_root_ang_vel_b_from_qvel,
    current_root_pose_from_qpos,
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
    settle_condition_mask,
    update_success_tracking,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


def excess_pitch(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    limit_rad: float = torch.pi * 80.0 / 180.0,
) -> torch.Tensor:
    """Terminate when the vehicle pitch leaves the allowed recovery envelope."""
    get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    _roll_rad, pitch_rad, _yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
    return pitch_rad.abs() > float(limit_rad)


def excess_depth_error(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    limit_m: float = 1.0,
) -> torch.Tensor:
    """Terminate when depth drifts too far from the reset reference."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    depth_error_m = root_pos_w[:, 2] - state.z_ref_m
    return depth_error_m.abs() > float(limit_m)


def excess_xy_drift(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
    limit_m: float = 1.0,
) -> torch.Tensor:
    """Terminate when world-frame XY drift exceeds the allowed radius."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    xy_error_w = root_pos_w[:, :2] - state.xy_ref_w
    xy_drift_m = torch.linalg.vector_norm(xy_error_w, dim=1)
    return xy_drift_m > float(limit_m)


def roll_task_success(
    env: ManagerBasedRlEnv,
    *,
    target_roll_rad: float,
    roll_direction: int,
    settle_steps: int,
    entity_name: str = "robot",
    settle_pitch_limit_rad: float = torch.pi * 10.0 / 180.0,
    settle_yaw_limit_rad: float = torch.pi * 15.0 / 180.0,
    settle_ang_vel_limit_rad_s: float = 0.25,
    settle_depth_error_limit_m: float = 0.15,
) -> torch.Tensor:
    """Terminate with success after the target is reached and held while settled."""
    state = get_roll_task_state(env, entity_name=entity_name)
    robot: Entity = env.scene[entity_name]

    current_step = env.episode_length_buf
    eval_mask = state.last_success_eval_step != current_step
    eval_env_ids = eval_mask.nonzero(as_tuple=False).squeeze(-1)
    if eval_env_ids.numel() > 0:
        root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
        _roll_rad, pitch_rad, yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
        depth_error_m = root_pos_w[:, 2] - state.z_ref_m
        yaw_error_rad = wrap_to_pi(yaw_rad - state.psi_ref_rad)
        settle_mask = settle_condition_mask(
            pitch_rad=pitch_rad,
            yaw_error_rad=yaw_error_rad,
            ang_vel_b_rad_s=current_root_ang_vel_b_from_qvel(robot),
            depth_error_m=depth_error_m,
            pitch_limit_rad=settle_pitch_limit_rad,
            yaw_limit_rad=settle_yaw_limit_rad,
            ang_vel_limit_rad_s=settle_ang_vel_limit_rad_s,
            depth_error_limit_m=settle_depth_error_limit_m,
        )
        next_target_reached, next_settle_counter, _success = update_success_tracking(
            phi_total_rad=state.phi_total_rad,
            target_reached=state.target_reached,
            settle_counter_steps=state.settle_counter_steps,
            settle_mask=settle_mask,
            target_roll_rad=target_roll_rad,
            roll_direction=roll_direction,
            settle_steps=settle_steps,
        )

        state.target_reached[eval_env_ids] = next_target_reached[eval_env_ids]
        state.settle_counter_steps[eval_env_ids] = next_settle_counter[eval_env_ids]
        state.last_success_eval_step[eval_env_ids] = current_step[eval_env_ids]

    return state.settle_counter_steps >= int(settle_steps)


__all__ = [
    "excess_depth_error",
    "excess_pitch",
    "excess_xy_drift",
    "roll_task_success",
]
