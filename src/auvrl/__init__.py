"""AUVRL package."""

from .actuator.body_wrench_action import BodyWrenchAction, BodyWrenchActionCfg
from .actuator.thruster_actuator import (
    THRUSTER_LOCAL_AXIS,
    ThrusterActuator,
    ThrusterActuatorCfg,
)
from .config.auv_cfg import AUVMjlabCfg
from .config.thruster_cfg import ThrusterModelCfg
from .envs.taluy_env_cfg import make_taluy_auv_env_cfg, make_taluy_base_env_cfg
from .envs.events import (
    randomize_thruster_supply_voltage,
    randomize_water_current_velocity,
)
from .sim.hydrodynamics import AUVBodyState, HydroConfig, HydrodynamicsModel
from .sim.underwater_hydro_action import UnderwaterHydroAction, UnderwaterHydroActionCfg
from .tasks.roll import (
    ROLL_CURRICULUM_STAGES,
    RollCurriculumStage,
    get_roll_curriculum_stage,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from .tasks.velocity import (
    UniformBodyVelocityCommand,
    UniformBodyVelocityCommandCfg,
    make_taluy_velocity_env_cfg,
    taluy_velocity_ppo_runner_cfg,
)

__all__ = [
    "AUVBodyState",
    "AUVMjlabCfg",
    "BodyWrenchAction",
    "BodyWrenchActionCfg",
    "HydroConfig",
    "HydrodynamicsModel",
    "ROLL_CURRICULUM_STAGES",
    "RollCurriculumStage",
    "THRUSTER_LOCAL_AXIS",
    "ThrusterActuator",
    "ThrusterActuatorCfg",
    "ThrusterModelCfg",
    "UnderwaterHydroAction",
    "UnderwaterHydroActionCfg",
    "make_taluy_roll_env_cfg",
    "get_roll_curriculum_stage",
    "UniformBodyVelocityCommand",
    "UniformBodyVelocityCommandCfg",
    "make_taluy_base_env_cfg",
    "make_taluy_auv_env_cfg",
    "make_taluy_velocity_env_cfg",
    "randomize_thruster_supply_voltage",
    "randomize_water_current_velocity",
    "taluy_roll_ppo_runner_cfg",
    "taluy_velocity_ppo_runner_cfg",
]
