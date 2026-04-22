"""Taluy deterministic roll-task environment configuration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from auvrl.envs.taluy_env_cfg import make_taluy_base_env_cfg
from auvrl.tasks.roll.roll_env_cfg import make_roll_env_cfg


def make_taluy_roll_env_cfg(
    *,
    num_envs: int = 1,
    target_roll_deg: float = 720.0,
    roll_direction: int = 1,
    episode_length_s: float = 20.0,
    settle_window_s: float = 1.0,
) -> ManagerBasedRlEnvCfg:
    """Create the Taluy v1 roll task with nominal physics and body-wrench control."""
    robot_base_env_cfg = make_taluy_base_env_cfg(action_space="body_wrench")
    cfg = make_roll_env_cfg(
        robot_base_env_cfg=robot_base_env_cfg,
        target_roll_deg=target_roll_deg,
        roll_direction=roll_direction,
        settle_window_s=settle_window_s,
    )
    cfg.scene.num_envs = num_envs
    cfg.episode_length_s = episode_length_s
    return cfg


__all__ = ["make_taluy_roll_env_cfg"]
