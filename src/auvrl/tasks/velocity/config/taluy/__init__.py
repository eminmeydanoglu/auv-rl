"""Taluy velocity task configurations."""

from .env_cfgs import make_taluy_velocity_env_cfg
from .rl_cfg import taluy_velocity_ppo_runner_cfg

__all__ = [
    "make_taluy_velocity_env_cfg",
    "taluy_velocity_ppo_runner_cfg",
]
