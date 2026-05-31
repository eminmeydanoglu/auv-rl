"""Taluy deterministic roll-task environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from auvrl.envs.taluy_env_cfg import make_taluy_base_env_cfg
from auvrl.tasks.roll.auto_curriculum import (
    POST_C3L_POLISH_AUTO_CURRICULUM,
    POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM,
    build_post_c3l_polish_curriculum,
    build_post_c3r_settle_saturation_curriculum,
)
from auvrl.tasks.roll.curriculum import get_roll_curriculum_stage
from auvrl.tasks.roll.mdp import nonroll_body_wrench_action_rate_l2
from auvrl.tasks.roll.roll_env_cfg import make_roll_env_cfg


def make_taluy_roll_env_cfg(
    *,
    num_envs: int = 1,
    curriculum_stage: str | None = None,
    target_roll_deg: float = 720.0,
    roll_direction: int = 1,
    episode_length_s: float | None = None,
    settle_window_s: float = 1.0,
    auto_curriculum: str | None = None,
    auto_curriculum_goal_stage: str = "c3q_720_c3l_strict_settle",
) -> ManagerBasedRlEnvCfg:
    """Create the Taluy v1 roll task with nominal physics and body-wrench control."""
    if auto_curriculum is not None:
        supported_auto_curricula = {
            POST_C3L_POLISH_AUTO_CURRICULUM,
            POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM,
        }
        if auto_curriculum not in supported_auto_curricula:
            raise ValueError(f"Unsupported roll auto curriculum: {auto_curriculum}.")
        if auto_curriculum == POST_C3L_POLISH_AUTO_CURRICULUM and curriculum_stage is None:
            curriculum_stage = "c3l_720_xy_guard"
        elif (
            auto_curriculum == POST_C3L_POLISH_AUTO_CURRICULUM
            and curriculum_stage != "c3l_720_xy_guard"
        ):
            raise ValueError(
                "post_c3l_polish auto curriculum requires "
                "curriculum_stage='c3l_720_xy_guard'."
            )
        if (
            auto_curriculum == POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM
            and curriculum_stage is None
        ):
            curriculum_stage = "c3r_720_post_target_tx_brake"
            if auto_curriculum_goal_stage == "c3q_720_c3l_strict_settle":
                auto_curriculum_goal_stage = "c3t_720_c3r_settle_sat_guard"
        elif (
            auto_curriculum == POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM
            and curriculum_stage != "c3r_720_post_target_tx_brake"
        ):
            raise ValueError(
                "post_c3r_settle_saturation auto curriculum requires "
                "curriculum_stage='c3r_720_post_target_tx_brake'."
            )
        elif (
            auto_curriculum == POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM
            and auto_curriculum_goal_stage == "c3q_720_c3l_strict_settle"
        ):
            auto_curriculum_goal_stage = "c3t_720_c3r_settle_sat_guard"

    roll_kwargs = {
        "target_roll_deg": target_roll_deg,
        "roll_direction": roll_direction,
        "settle_window_s": settle_window_s,
    }
    stage = None
    if curriculum_stage is not None:
        stage = get_roll_curriculum_stage(curriculum_stage)
        roll_kwargs.update(stage.roll_env_kwargs())
        roll_kwargs["roll_direction"] = roll_direction
        if auto_curriculum == POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM:
            goal_stage = get_roll_curriculum_stage(auto_curriculum_goal_stage)
            roll_kwargs.update(
                {
                    "excess_pitch_deg": goal_stage.excess_pitch_deg,
                    "excess_depth_error_m": goal_stage.excess_depth_error_m,
                    "excess_xy_drift_m": goal_stage.excess_xy_drift_m,
                    "settle_pitch_limit_deg": goal_stage.settle_pitch_limit_deg,
                    "settle_yaw_limit_deg": goal_stage.settle_yaw_limit_deg,
                    "settle_depth_error_limit_m": (
                        goal_stage.settle_depth_error_limit_m
                    ),
                    "settle_xy_drift_limit_m": goal_stage.settle_xy_drift_limit_m,
                }
            )
        if episode_length_s is None:
            episode_length_s = stage.episode_length_s

    if episode_length_s is None:
        episode_length_s = 20.0

    robot_base_env_cfg = make_taluy_base_env_cfg(action_space="body_wrench")
    cfg = make_roll_env_cfg(
        robot_base_env_cfg=robot_base_env_cfg,
        **roll_kwargs,
    )
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_length_s
    if auto_curriculum == POST_C3L_POLISH_AUTO_CURRICULUM:
        if stage is None:
            stage = get_roll_curriculum_stage("c3l_720_xy_guard")
        goal_stage = get_roll_curriculum_stage(auto_curriculum_goal_stage)
        cfg.curriculum.update(
            build_post_c3l_polish_curriculum(
                start_stage=stage,
                goal_stage=goal_stage,
            )
        )
    if auto_curriculum == POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM:
        if stage is None:
            stage = get_roll_curriculum_stage("c3r_720_post_target_tx_brake")
        goal_stage = get_roll_curriculum_stage(auto_curriculum_goal_stage)
        if "nonroll_wrench_rate" not in cfg.rewards:
            cfg.rewards["nonroll_wrench_rate"] = RewardTermCfg(
                func=nonroll_body_wrench_action_rate_l2,
                weight=-stage.k_nonroll_wrench_rate,
                params={"action_name": "body_wrench"},
            )
        cfg.curriculum.update(
            build_post_c3r_settle_saturation_curriculum(
                start_stage=stage,
                goal_stage=goal_stage,
            )
        )
    return cfg


__all__ = ["make_taluy_roll_env_cfg"]
