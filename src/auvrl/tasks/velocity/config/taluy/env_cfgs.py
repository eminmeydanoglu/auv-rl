"""Taluy velocity environment configurations."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from auvrl.envs.taluy_env_cfg import EventMode, make_taluy_base_env_cfg
from auvrl.tasks.velocity import mdp
from auvrl.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from auvrl.utils.observation import obs_scale_from_range


def make_taluy_velocity_env_cfg(
    *,
    # env settings
    num_envs: int = 1,
    episode_length_s: float = 30.0,
    #
    # command settings
    command_resampling_time_s: tuple[float, float] = (2.0, 5.0),
    command_rel_zero_envs: float = 0.1,
    command_init_velocity_prob: float = 0.0,
    command_lin_vel_x_range_m_s: tuple[float, float] = (-0.5, 0.5),
    command_lin_vel_y_range_m_s: tuple[float, float] = (-0.5, 0.5),
    command_lin_vel_z_range_m_s: tuple[float, float] = (-0.4, 0.4),
    command_ang_vel_x_range_rad_s: tuple[float, float] = (-1.0, 1.0),
    command_ang_vel_y_range_rad_s: tuple[float, float] = (-1.0, 1.0),
    command_ang_vel_z_range_rad_s: tuple[float, float] = (-1.2, 1.2),
    #
    # event parameters
    thruster_voltage_event_mode: EventMode = "disabled",
    thruster_voltage_range_v: tuple[float, float] = (16.0, 16.0),
    current_event_mode: EventMode = "disabled",
    current_speed_range_m_s: tuple[float, float] = (0.0, 0.0),
    current_yaw_range_rad: tuple[float, float] = (0.0, 0.0),
    vertical_current_range_m_s: tuple[float, float] = (0.0, 0.0),
) -> ManagerBasedRlEnvCfg:
    """Create the Taluy 6-DOF velocity task with body-wrench actions."""

    # vehicle base env cfg

    robot_base_env_cfg = make_taluy_base_env_cfg(
        action_space="body_wrench",
        thruster_voltage_event_mode=thruster_voltage_event_mode,
        thruster_voltage_range_v=thruster_voltage_range_v,
        current_event_mode=current_event_mode,
        current_speed_range_m_s=current_speed_range_m_s,
        current_yaw_range_rad=current_yaw_range_rad,
        vertical_current_range_m_s=vertical_current_range_m_s,
    )

    # task base env cfg

    cfg = make_velocity_env_cfg(
        robot_base_env_cfg=robot_base_env_cfg,
        command_lin_vel_x_range_m_s=command_lin_vel_x_range_m_s,
        command_lin_vel_y_range_m_s=command_lin_vel_y_range_m_s,
        command_lin_vel_z_range_m_s=command_lin_vel_z_range_m_s,
        command_ang_vel_x_range_rad_s=command_ang_vel_x_range_rad_s,
        command_ang_vel_y_range_rad_s=command_ang_vel_y_range_rad_s,
        command_ang_vel_z_range_rad_s=command_ang_vel_z_range_rad_s,
        command_resampling_time_s=command_resampling_time_s,
        command_rel_zero_envs=command_rel_zero_envs,
        command_init_velocity_prob=command_init_velocity_prob,
    )

    # run specific overrides

    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_length_s

    # taluy-specific observations

    lin_obs_scale = (
        obs_scale_from_range(command_lin_vel_x_range_m_s),
        obs_scale_from_range(command_lin_vel_y_range_m_s),
        obs_scale_from_range(command_lin_vel_z_range_m_s),
    )

    # cfg.observations["actor"].terms["thruster_force_state"] = ObservationTermCfg(
    #     func=mdp.thruster_force_state,
    # )

    # critic gets everything the actor sees, plus privileged terms.
    # cfg.observations["critic"].terms["thruster_force_state"] = ObservationTermCfg(
    #     func=mdp.thruster_force_state,
    # )
    cfg.observations["critic"].terms["depth_error"] = ObservationTermCfg(
        func=mdp.depth_error,
        scale=0.5,
    )
    if current_event_mode != "disabled":
        cfg.observations["critic"].terms["current_velocity_b"] = ObservationTermCfg(
            func=mdp.current_velocity_b,
            scale=lin_obs_scale,
        )

    if thruster_voltage_event_mode != "disabled":
        cfg.observations["critic"].terms["thruster_voltage_offset"] = (
            ObservationTermCfg(
                func=mdp.thruster_voltage_offset,
            )
        )

    # taluy-specific rewards

    cfg.rewards["action_l2"] = RewardTermCfg(
        func=mdp.body_wrench_action_l2,
        weight=-2.0e-3,
        params={"action_name": "body_wrench"},
    )
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.body_wrench_action_rate_l2,
        weight=-1.0e-2,
        params={"action_name": "body_wrench"},
    )
    cfg.rewards["saturation"] = RewardTermCfg(
        func=mdp.body_wrench_saturation_penalty,
        weight=-5.0e-2,
        params={"action_name": "body_wrench"},
    )

    return cfg
