"""Taluy deterministic roll-task environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from auvrl.envs.taluy_env_cfg import make_taluy_base_env_cfg
from auvrl.tasks.roll.curriculum import get_roll_curriculum_stage
from auvrl.tasks.roll.roll_env_cfg import make_roll_env_cfg


def make_taluy_roll_env_cfg(
    *,
    num_envs: int = 1,
    curriculum_stage: str | None = None,
    target_roll_deg: float = 720.0,
    roll_direction: int = 1,
    episode_length_s: float | None = None,
    settle_window_s: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    """Create the Taluy v1 roll task with nominal physics and body-wrench control."""
    roll_kwargs = {
        "target_roll_deg": target_roll_deg,
        "roll_direction": roll_direction,
        "settle_window_s": settle_window_s,
    }
    if curriculum_stage is not None:
        stage = get_roll_curriculum_stage(curriculum_stage)
        roll_kwargs.update(stage.roll_env_kwargs())
        roll_kwargs["roll_direction"] = roll_direction
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
    return cfg


__all__ = ["make_taluy_roll_env_cfg"]
