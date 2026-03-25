"""Taluy base environment config builder."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mujoco

from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import SiteEffortActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.viewer import ViewerConfig

from auvrl.actuator.body_wrench_action import BodyWrenchActionCfg
from auvrl.actuator.thruster_actuator import make_thruster_actuator_cfg
from auvrl.config.auv_cfg import TALUY_CFG_PATH, load_auv_cfg
from auvrl.config.thruster_cfg import (
    THRUSTER_CFG_DIR,
    load_thruster_cfg,
)
from auvrl.envs.events import (
    randomize_thruster_supply_voltage,
    randomize_water_current_velocity,
)
from auvrl.sim.underwater_hydro_action import make_underwater_hydro_action_cfg

EventMode = Literal["disabled", "startup", "reset"]
TaluyActionSpace = Literal["thruster", "body_wrench"]

_AUVRL_ROOT = Path(__file__).resolve().parents[1]
_TALUY_THRUSTER_PATTERN = ("thruster_[0-7]_site",)


def _taluy_spec():
    taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)
    taluy_xml_path = _AUVRL_ROOT / taluy_cfg.model_xml_path
    mj_spec = getattr(mujoco, "MjSpec")
    spec = mj_spec.from_file(str(taluy_xml_path))
    spec.meshdir = str(taluy_xml_path.parent)
    return spec


def make_taluy_base_env_cfg(
    *,
    action_space: TaluyActionSpace = "body_wrench",
    thruster_voltage_event_mode: EventMode = "disabled",
    thruster_voltage_range_v: tuple[float, float] = (16.0, 16.0),
    current_event_mode: EventMode = "disabled",
    current_speed_range_m_s: tuple[float, float] = (0.0, 0.0),
    current_yaw_range_rad: tuple[float, float] = (0.0, 0.0),
    vertical_current_range_m_s: tuple[float, float] = (0.0, 0.0),
) -> ManagerBasedRlEnvCfg:
    """Create the base Taluy MJLab environment config"""

    taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)
    thruster_cfg = load_thruster_cfg(
        THRUSTER_CFG_DIR / f"{taluy_cfg.thruster_model}.yaml"
    )

    ##
    # Scene
    ##

    scene = SceneCfg(
        num_envs=1,
        entities={
            "robot": EntityCfg(
                spec_fn=_taluy_spec,
                articulation=EntityArticulationInfoCfg(
                    actuators=(
                        make_thruster_actuator_cfg(
                            target_names_expr=_TALUY_THRUSTER_PATTERN,
                            thruster_cfg=thruster_cfg,
                        ),
                    ),
                ),
            ),
        },
    )

    ##
    # Actions
    ##

    actions: dict[str, ActionTermCfg] = {
        "hydro": make_underwater_hydro_action_cfg(
            auv_cfg=taluy_cfg,
            entity_name="robot",
        ),
    }

    if action_space == "thruster":
        actions["thrusters"] = SiteEffortActionCfg(
            entity_name="robot",
            actuator_names=_TALUY_THRUSTER_PATTERN,
            scale=1.0,
            offset=0.0,
            preserve_order=True,
        )

    if action_space == "body_wrench":
        actions["body_wrench"] = BodyWrenchActionCfg(
            entity_name="robot",
            body_name=taluy_cfg.body_name,
            actuator_names=taluy_cfg.thruster_site_names,
            wrench_limit=taluy_cfg.body_wrench_limit,
            preserve_order=True,
            neutralize_com_coupling=True,
            require_full_rank=True,
            site_force_limit_n=thruster_cfg.command_limit,
        )
    elif action_space != "thruster":
        raise ValueError(
            f"action_space must be 'thruster' or 'body_wrench'. Got '{action_space}'."
        )

    ##
    # Events
    ##

    events: dict[str, EventTermCfg] = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default,
            mode="reset",
        ),
    }

    if thruster_voltage_event_mode != "disabled":
        events["randomize_thruster_supply_voltage"] = EventTermCfg(
            func=randomize_thruster_supply_voltage,
            mode=thruster_voltage_event_mode,
            params={
                "entity_name": "robot",
                "voltage_range": thruster_voltage_range_v,
            },
        )

    if current_event_mode != "disabled":
        events["randomize_water_current_velocity"] = EventTermCfg(
            func=randomize_water_current_velocity,
            mode=current_event_mode,
            params={
                "action_term_name": "hydro",
                "speed_range_m_s": current_speed_range_m_s,
                "yaw_range_rad": current_yaw_range_rad,
                "vertical_range_m_s": vertical_current_range_m_s,
            },
        )

    ##
    # Terminations
    ##

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "nan_detected": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    ##
    # Viewer
    ##

    viewer = ViewerConfig(
        origin_type=ViewerConfig.OriginType.ASSET_BODY,
        entity_name="robot",
        body_name=taluy_cfg.body_name,
        distance=taluy_cfg.viewer_distance,
        elevation=taluy_cfg.viewer_elevation,
        azimuth=taluy_cfg.viewer_azimuth,
    )

    ##
    # Simulation
    ##

    sim = SimulationCfg(
        mujoco=MujocoCfg(
            timestep=0.002,
            gravity=(0.0, 0.0, 0.0),
            iterations=10,
            ls_iterations=20,
        )
    )

    ##
    # Assemble and return
    ##

    return ManagerBasedRlEnvCfg(
        scene=scene,
        actions=actions,
        events=events,
        terminations=terminations,
        viewer=viewer,
        sim=sim,
        decimation=4,
        episode_length_s=20.0,
    )


make_taluy_auv_env_cfg = make_taluy_base_env_cfg


__all__ = [
    "EventMode",
    "TaluyActionSpace",
    "make_taluy_base_env_cfg",
    "make_taluy_auv_env_cfg",
]
