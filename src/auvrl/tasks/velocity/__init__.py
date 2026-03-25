"""Velocity-task scaffolding for AUV body-velocity control."""

from .config.taluy import make_taluy_velocity_env_cfg, taluy_velocity_ppo_runner_cfg
from .mdp import UniformBodyVelocityCommand, UniformBodyVelocityCommandCfg
from .velocity_env_cfg import make_velocity_env_cfg

__all__ = [
    "UniformBodyVelocityCommand",
    "UniformBodyVelocityCommandCfg",
    "make_velocity_env_cfg",
    "make_taluy_velocity_env_cfg",
    "taluy_velocity_ppo_runner_cfg",
]
