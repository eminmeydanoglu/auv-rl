"""Runtime state and pure helpers for the Taluy roll task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import wrap_to_pi

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv


@dataclass
class RollTaskState:
    """Per-environment task state shared by roll observations/rewards/terms."""

    initialized: torch.Tensor
    last_update_step: torch.Tensor
    last_success_eval_step: torch.Tensor
    phi_total_rad: torch.Tensor
    delta_roll_rad: torch.Tensor
    prev_roll_rad: torch.Tensor
    z_ref_m: torch.Tensor
    psi_ref_rad: torch.Tensor
    xy_ref_w: torch.Tensor
    target_reached: torch.Tensor
    settle_counter_steps: torch.Tensor


def _ensure_roll_task_state(env: ManagerBasedRlEnv) -> RollTaskState:
    """Create the shared roll-task state buffers on first access."""
    state = getattr(env, "_auvrl_roll_task_state", None)
    if state is None:
        state = RollTaskState(
            initialized=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
            last_update_step=torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
            last_success_eval_step=torch.full(
                (env.num_envs,),
                fill_value=-1,
                dtype=torch.long,
                device=env.device,
            ),
            phi_total_rad=torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
            delta_roll_rad=torch.zeros(
                env.num_envs, dtype=torch.float, device=env.device
            ),
            prev_roll_rad=torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
            z_ref_m=torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
            psi_ref_rad=torch.zeros(env.num_envs, dtype=torch.float, device=env.device),
            xy_ref_w=torch.zeros(env.num_envs, 2, dtype=torch.float, device=env.device),
            target_reached=torch.zeros(
                env.num_envs, dtype=torch.bool, device=env.device
            ),
            settle_counter_steps=torch.zeros(
                env.num_envs, dtype=torch.long, device=env.device
            ),
        )
        setattr(env, "_auvrl_roll_task_state", state)
    return state


def _selected_env_ids(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
) -> torch.Tensor:
    """Normalize optional env selection into a dense 1-D index tensor."""
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    return env_ids.to(device=env.device, dtype=torch.long)


def quat_wxyz_to_roll_pitch_yaw(
    quat_wxyz: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a batch of quaternions in ``(w, x, y, z)`` to roll/pitch/yaw."""
    quat = quat_wxyz / quat_wxyz.norm(dim=-1, keepdim=True).clamp_min(1.0e-9)
    w, x, y, z = quat.unbind(dim=-1)

    sin_roll = 2.0 * (w * x + y * z)
    cos_roll = 1.0 - 2.0 * (x * x + y * y)
    roll = torch.atan2(sin_roll, cos_roll)

    sin_pitch = 2.0 * (w * y - z * x)
    pitch = torch.asin(torch.clamp(sin_pitch, min=-1.0, max=1.0))

    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(sin_yaw, cos_yaw)

    return roll, pitch, yaw


def current_root_pose_from_qpos(entity: Entity) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the root position and quaternion directly from ``qpos``."""
    if not hasattr(entity.data, "data") or not hasattr(entity.data, "indexing"):
        return entity.data.root_link_pos_w, entity.data.root_link_quat_w
    qpos = entity.data.data.qpos[:, entity.data.indexing.free_joint_q_adr]
    return qpos[:, :3], qpos[:, 3:7]


def current_root_ang_vel_b_from_qvel(entity: Entity) -> torch.Tensor:
    """Return the root angular velocity directly from free-joint ``qvel``."""
    if not hasattr(entity.data, "data") or not hasattr(entity.data, "indexing"):
        return entity.data.root_link_ang_vel_b
    qvel = entity.data.data.qvel[:, entity.data.indexing.free_joint_v_adr]
    return qvel[:, 3:6]


def action_term_slice(
    env: ManagerBasedRlEnv,
    action_name: str,
) -> slice:
    """Return the flat action slice for a named action term."""
    start = 0
    for name, dim in zip(
        env.action_manager.active_terms,
        env.action_manager.action_term_dim,
        strict=False,
    ):
        stop = start + dim
        if name == action_name:
            return slice(start, stop)
        start = stop
    raise ValueError(f"Action term '{action_name}' not found.")


def update_phi_total(
    phi_total_rad: torch.Tensor,
    prev_roll_rad: torch.Tensor,
    current_roll_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the unwrapped roll counter from the previous and current roll."""
    delta_roll_rad = wrap_to_pi(current_roll_rad - prev_roll_rad)
    return phi_total_rad + delta_roll_rad, current_roll_rad, delta_roll_rad


def normalized_phi_total(
    phi_total_rad: torch.Tensor,
    *,
    target_roll_rad: float,
    roll_direction: int,
    clip_limit: float = 2.0,
) -> torch.Tensor:
    """Normalize signed roll progress against the task target."""
    if target_roll_rad <= 0.0:
        raise ValueError(f"target_roll_rad must be positive, got {target_roll_rad}.")
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    if clip_limit <= 0.0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}.")

    signed_progress = (float(roll_direction) * phi_total_rad) / float(target_roll_rad)
    return torch.clamp(signed_progress, min=-clip_limit, max=clip_limit)


def settle_condition_mask(
    *,
    pitch_rad: torch.Tensor,
    yaw_error_rad: torch.Tensor,
    ang_vel_b_rad_s: torch.Tensor,
    depth_error_m: torch.Tensor,
    pitch_limit_rad: float,
    yaw_limit_rad: float,
    ang_vel_limit_rad_s: float,
    depth_error_limit_m: float,
) -> torch.Tensor:
    """Return the per-env mask for the success settle window constraints."""
    ang_speed = torch.linalg.vector_norm(ang_vel_b_rad_s, dim=1)
    return (
        (pitch_rad.abs() <= float(pitch_limit_rad))
        & (yaw_error_rad.abs() <= float(yaw_limit_rad))
        & (ang_speed <= float(ang_vel_limit_rad_s))
        & (depth_error_m.abs() <= float(depth_error_limit_m))
    )


def update_success_tracking(
    *,
    phi_total_rad: torch.Tensor,
    target_reached: torch.Tensor,
    settle_counter_steps: torch.Tensor,
    settle_mask: torch.Tensor,
    target_roll_rad: float,
    roll_direction: int,
    settle_steps: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance target-reached and settle-window state, then emit success."""
    if target_roll_rad <= 0.0:
        raise ValueError(f"target_roll_rad must be positive, got {target_roll_rad}.")
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    if settle_steps <= 0:
        raise ValueError(f"settle_steps must be positive, got {settle_steps}.")

    reached_now = float(roll_direction) * phi_total_rad >= float(target_roll_rad)
    next_target_reached = target_reached | reached_now
    in_settle_window = next_target_reached & settle_mask
    next_settle_counter = torch.where(
        in_settle_window,
        settle_counter_steps + 1,
        torch.zeros_like(settle_counter_steps),
    )
    success = next_settle_counter >= int(settle_steps)
    return next_target_reached, next_settle_counter, success


def get_roll_task_state(
    env: ManagerBasedRlEnv,
    entity_name: str = "robot",
) -> RollTaskState:
    """Return the shared roll-task state, lazily initializing/resetting it."""
    state = _ensure_roll_task_state(env)
    if not torch.all(state.initialized):
        uninitialized_env_ids = (~state.initialized).nonzero(as_tuple=False).squeeze(-1)
        reset_roll_task_state(env, uninitialized_env_ids, entity_name=entity_name)

    robot: Entity = env.scene[entity_name]
    episode_steps = env.episode_length_buf

    update_mask = state.initialized & (episode_steps > state.last_update_step)
    update_env_ids = update_mask.nonzero(as_tuple=False).squeeze(-1)
    if update_env_ids.numel() > 0:
        _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
        current_roll_rad, _pitch_rad, _yaw_rad = quat_wxyz_to_roll_pitch_yaw(
            quat_wxyz[update_env_ids]
        )
        next_phi_total, next_prev_roll, delta_roll = update_phi_total(
            state.phi_total_rad[update_env_ids],
            state.prev_roll_rad[update_env_ids],
            current_roll_rad,
        )
        state.phi_total_rad[update_env_ids] = next_phi_total
        state.prev_roll_rad[update_env_ids] = next_prev_roll
        state.delta_roll_rad[update_env_ids] = delta_roll
        state.last_update_step[update_env_ids] = episode_steps[update_env_ids]

    return state


def reset_roll_task_state(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    entity_name: str = "robot",
) -> None:
    """Latch reset-time references and clear task progress for selected envs."""
    state = _ensure_roll_task_state(env)
    selected_env_ids = _selected_env_ids(env, env_ids)
    if selected_env_ids.numel() == 0:
        return

    robot: Entity = env.scene[entity_name]
    root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
    quat_wxyz = quat_wxyz[selected_env_ids]
    root_pos_w = root_pos_w[selected_env_ids]
    roll_rad, _pitch_rad, yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)

    state.initialized[selected_env_ids] = True
    # Reset events run before MJLab clears episode_length_buf back to zero.
    # The roll integrator should restart from step 0 for every fresh episode.
    state.last_update_step[selected_env_ids] = 0
    state.last_success_eval_step[selected_env_ids] = -1
    state.phi_total_rad[selected_env_ids] = 0.0
    state.delta_roll_rad[selected_env_ids] = 0.0
    state.prev_roll_rad[selected_env_ids] = roll_rad
    state.z_ref_m[selected_env_ids] = root_pos_w[:, 2]
    state.psi_ref_rad[selected_env_ids] = yaw_rad
    state.xy_ref_w[selected_env_ids] = root_pos_w[:, :2]
    state.target_reached[selected_env_ids] = False
    state.settle_counter_steps[selected_env_ids] = 0


__all__ = [
    "action_term_slice",
    "current_root_ang_vel_b_from_qvel",
    "current_root_pose_from_qpos",
    "RollTaskState",
    "get_roll_task_state",
    "normalized_phi_total",
    "quat_wxyz_to_roll_pitch_yaw",
    "reset_roll_task_state",
    "settle_condition_mask",
    "update_phi_total",
    "update_success_tracking",
]
