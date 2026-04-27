"""Base configuration for the deterministic v1 roll task."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from . import mdp
from .runtime import reset_roll_task_state


def make_roll_env_cfg(
    *,
    robot_base_env_cfg: ManagerBasedRlEnvCfg,
    target_roll_deg: float = 720.0,
    roll_direction: int = 1,
    settle_window_s: float = 1.0,
    k_prog: float = 8.0,
    k_xy: float = 1.5,
    k_pitch: float = 1.0,
    k_yaw: float = 0.5,
    k_depth: float = 1.0,
    k_smooth: float = 0.01,
    excess_pitch_deg: float = 80.0,
    excess_depth_error_m: float = 1.0,
    excess_xy_drift_m: float = 1.0,
    settle_pitch_limit_deg: float = 10.0,
    settle_yaw_limit_deg: float = 15.0,
    settle_ang_vel_limit_rad_s: float = 0.25,
    settle_depth_error_limit_m: float = 0.15,
    terminal_success_weight: float = 100.0,
    terminal_failure_weight: float = -50.0,
) -> ManagerBasedRlEnvCfg:
    """Create the base deterministic roll-task config."""
    if target_roll_deg <= 0.0:
        raise ValueError(f"target_roll_deg must be positive, got {target_roll_deg}.")
    if roll_direction not in (-1, 1):
        raise ValueError(f"roll_direction must be +/-1, got {roll_direction}.")
    if settle_window_s <= 0.0:
        raise ValueError(f"settle_window_s must be positive, got {settle_window_s}.")

    cfg = robot_base_env_cfg
    cfg.commands = {}
    cfg.scale_rewards_by_dt = False

    target_roll_rad = math.radians(target_roll_deg)
    step_dt = cfg.sim.mujoco.timestep * cfg.decimation
    settle_steps = max(1, math.ceil(settle_window_s / step_dt))

    actor_terms = {
        "base_quat_wxyz": ObservationTermCfg(func=mdp.base_quat_wxyz),
        "base_ang_vel_b": ObservationTermCfg(func=mdp.base_ang_vel),
        "depth_error": ObservationTermCfg(func=mdp.depth_error_from_ref),
        "phi_total_norm": ObservationTermCfg(
            func=mdp.phi_total_norm,
            params={
                "target_roll_rad": target_roll_rad,
                "roll_direction": roll_direction,
            },
        ),
        "last_action": ObservationTermCfg(func=mdp.last_body_wrench_action),
    }
    critic_terms = {
        **actor_terms,
        "base_lin_vel_b": ObservationTermCfg(func=mdp.base_lin_vel),
        "xy_error_w": ObservationTermCfg(func=mdp.xy_error_w),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_terms,
            concatenate_terms=True,
            enable_corruption=False,
        ),
    }

    # Keep task state resets explicit instead of inferring them from episode counters.
    cfg.events["reset_roll_task_state"] = EventTermCfg(
        func=reset_roll_task_state,
        mode="reset",
    )

    cfg.rewards = {
        "roll_progress": RewardTermCfg(
            func=mdp.roll_progress,
            weight=k_prog,
            params={
                "roll_direction": roll_direction,
                "target_roll_rad": target_roll_rad,
            },
        ),
        "xy_drift": RewardTermCfg(
            func=mdp.xy_drift_penalty,
            weight=k_xy,
        ),
        "pitch_penalty": RewardTermCfg(
            func=mdp.pitch_penalty,
            weight=k_pitch,
        ),
        "yaw_hold": RewardTermCfg(
            func=mdp.yaw_hold_penalty,
            weight=k_yaw,
        ),
        "depth_hold": RewardTermCfg(
            func=mdp.depth_hold_penalty,
            weight=k_depth,
        ),
        "action_smoothness": RewardTermCfg(
            func=mdp.body_wrench_action_rate_l2,
            weight=-k_smooth,
            params={"action_name": "body_wrench"},
        ),
        "terminal_success": RewardTermCfg(
            func=mdp.terminal_success_reward,
            weight=terminal_success_weight,
            params={"termination_name": "task_success"},
        ),
        "terminal_failure": RewardTermCfg(
            func=mdp.terminal_failure_reward,
            weight=terminal_failure_weight,
            params={"success_term_name": "task_success"},
        ),
    }

    cfg.metrics = {
        "roll_progress_ratio_last": MetricsTermCfg(
            func=mdp.roll_progress_ratio,
            reduce="last",
            params={
                "target_roll_rad": target_roll_rad,
                "roll_direction": roll_direction,
            },
        ),
        "phi_total_rad_last": MetricsTermCfg(
            func=mdp.phi_total_rad,
            reduce="last",
        ),
        "target_reached_last": MetricsTermCfg(
            func=mdp.target_reached,
            reduce="last",
        ),
        "settle_counter_s_last": MetricsTermCfg(
            func=mdp.settle_counter_s,
            reduce="last",
        ),
        "depth_abs_error_m": MetricsTermCfg(func=mdp.depth_abs_error_m),
        "xy_drift_m": MetricsTermCfg(func=mdp.xy_drift_m),
        "pitch_abs_rad": MetricsTermCfg(func=mdp.pitch_abs_rad),
        "yaw_abs_error_rad": MetricsTermCfg(func=mdp.yaw_abs_error_rad),
        "root_ang_speed_rad_s": MetricsTermCfg(func=mdp.root_ang_speed_rad_s),
        "body_wrench_action_l2": MetricsTermCfg(func=mdp.body_wrench_action_l2),
        "body_wrench_saturation_fraction": MetricsTermCfg(
            func=mdp.body_wrench_saturation_fraction
        ),
        "water_current_speed_m_s": MetricsTermCfg(func=mdp.water_current_speed_m_s),
        "hydro_wrench_norm": MetricsTermCfg(func=mdp.hydro_wrench_norm),
    }

    cfg.terminations = {
        **cfg.terminations,
        "excess_pitch": TerminationTermCfg(
            func=mdp.excess_pitch,
            params={"limit_rad": math.radians(excess_pitch_deg)},
        ),
        "excess_depth_error": TerminationTermCfg(
            func=mdp.excess_depth_error,
            params={"limit_m": excess_depth_error_m},
        ),
        "excess_xy_drift": TerminationTermCfg(
            func=mdp.excess_xy_drift,
            params={"limit_m": excess_xy_drift_m},
        ),
        "task_success": TerminationTermCfg(
            func=mdp.roll_task_success,
            params={
                "target_roll_rad": target_roll_rad,
                "roll_direction": roll_direction,
                "settle_steps": settle_steps,
                "settle_pitch_limit_rad": math.radians(settle_pitch_limit_deg),
                "settle_yaw_limit_rad": math.radians(settle_yaw_limit_deg),
                "settle_ang_vel_limit_rad_s": settle_ang_vel_limit_rad_s,
                "settle_depth_error_limit_m": settle_depth_error_limit_m,
            },
        ),
    }

    return cfg


__all__ = ["make_roll_env_cfg"]
