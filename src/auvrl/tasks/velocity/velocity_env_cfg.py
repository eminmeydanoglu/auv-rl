"""Velocity task base configuration.

This module provides a factory function to create a base 6-DOF velocity
tracking task config.  Vehicle-specific configurations (under ``config/``)
call this factory, then layer on actuator-specific observations, rewards,
and tuning.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from auvrl.utils.observation import obs_scale_from_range

from . import mdp


def make_velocity_env_cfg(
    *,
    robot_base_env_cfg: ManagerBasedRlEnvCfg,
    command_lin_vel_x_range_m_s: tuple[float, float],
    command_lin_vel_y_range_m_s: tuple[float, float],
    command_lin_vel_z_range_m_s: tuple[float, float],
    command_ang_vel_x_range_rad_s: tuple[float, float],
    command_ang_vel_y_range_rad_s: tuple[float, float],
    command_ang_vel_z_range_rad_s: tuple[float, float],
    command_resampling_time_s: tuple[float, float],
    command_rel_zero_envs: float,
    command_init_velocity_prob: float,
) -> ManagerBasedRlEnvCfg:
    """Create the base 6-DOF body-velocity tracking task.

    This sets up the command, core observations, and velocity-tracking
    rewards.  It does **not** add actuator-specific observations or
    rewards — those belong in the vehicle config layer.
    """

    cfg = robot_base_env_cfg

    # -- observation scales derived from command ranges --

    lin_obs_scale = (
        obs_scale_from_range(command_lin_vel_x_range_m_s),
        obs_scale_from_range(command_lin_vel_y_range_m_s),
        obs_scale_from_range(command_lin_vel_z_range_m_s),
    )
    ang_obs_scale = (
        obs_scale_from_range(command_ang_vel_x_range_rad_s),
        obs_scale_from_range(command_ang_vel_y_range_rad_s),
        obs_scale_from_range(command_ang_vel_z_range_rad_s),
    )
    command_obs_scale = lin_obs_scale + ang_obs_scale

    ##
    # Commands
    ##

    cfg.commands = {
        "body_velocity": mdp.UniformBodyVelocityCommandCfg(
            entity_name="robot",
            resampling_time_range=command_resampling_time_s,
            rel_zero_envs=command_rel_zero_envs,
            init_velocity_prob=command_init_velocity_prob,
            debug_vis=True,
            ranges=mdp.UniformBodyVelocityCommandCfg.Ranges(
                lin_vel_x=command_lin_vel_x_range_m_s,
                lin_vel_y=command_lin_vel_y_range_m_s,
                lin_vel_z=command_lin_vel_z_range_m_s,
                ang_vel_x=command_ang_vel_x_range_rad_s,
                ang_vel_y=command_ang_vel_y_range_rad_s,
                ang_vel_z=command_ang_vel_z_range_rad_s,
            ),
        )
    }

    ##
    # Observations
    ##

    actor_terms = {
        "base_lin_vel": ObservationTermCfg(
            func=envs_mdp.base_lin_vel,
            scale=lin_obs_scale,
        ),
        "base_ang_vel": ObservationTermCfg(
            func=envs_mdp.base_ang_vel,
            scale=ang_obs_scale,
        ),
        "projected_gravity": ObservationTermCfg(func=envs_mdp.projected_gravity),
        "command": ObservationTermCfg(
            func=envs_mdp.generated_commands,
            params={"command_name": "body_velocity"},
            scale=command_obs_scale,
        ),
        "last_action": ObservationTermCfg(
            func=envs_mdp.last_action,
            params={"action_name": "body_wrench"},
        ),
    }

    critic_terms = {**actor_terms}

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

    ##
    # Rewards
    ##

    cfg.rewards = {
        "track_linear_velocity": RewardTermCfg(
            func=mdp.track_body_linear_velocity,
            weight=3.0,
            params={
                "command_name": "body_velocity",
                "std": 0.30,
            },
        ),
        "track_angular_velocity": RewardTermCfg(
            func=mdp.track_body_angular_velocity,
            weight=2.5,
            params={
                "command_name": "body_velocity",
                "std": 0.50,
            },
        ),
    }

    return cfg
