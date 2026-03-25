"""MJLab AUV config models and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from auvrl.utils.type_aliases import Matrix6x6, Vector3, Vector6, Vector8, _ZERO_6X6


_AUVRL_ROOT = Path(__file__).resolve().parents[1]
TALUY_CFG_PATH = _AUVRL_ROOT / "asset_zoo" / "vehicles" / "taluy" / "taluy.yaml"
_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)
_PAIR_TOLERANCE = 1.0e-6


def _vector3_to_tuple(vector: np.ndarray) -> Vector3:
    values = np.asarray(vector, dtype=float).reshape(3)
    return (float(values[0]), float(values[1]), float(values[2]))


def _load_body_center_of_gravity(
    model_xml_path: Path,
    body_name: str,
) -> Vector3:
    mj_model = getattr(mujoco, "MjModel")
    mj_name2id = getattr(mujoco, "mj_name2id")
    mjt_obj = getattr(mujoco, "mjtObj")

    model = mj_model.from_xml_path(str(model_xml_path))
    body_id = mj_name2id(model, mjt_obj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in {model_xml_path}.")

    center_of_gravity_b_m = np.asarray(model.body_ipos[body_id], dtype=float).reshape(3)
    return _vector3_to_tuple(center_of_gravity_b_m)


class HydroYamlModel(BaseModel):
    model_config = _MODEL_CONFIG

    linear_damping_matrix: Matrix6x6
    quadratic_damping_matrix: Matrix6x6
    added_mass_6x6: Matrix6x6 = _ZERO_6X6

    current_world_m_s: Vector3 = (0.0, 0.0, 0.0)
    current_body_m_s: Vector3 | None = None

    fluid_density_kg_m3: float = Field(gt=0.0)
    gravity_m_s2: float = Field(gt=0.0)

    buoyancy_n: float | None = Field(default=None, ge=0.0)
    displaced_volume_m3: float | None = Field(default=None, ge=0.0)

    center_of_buoyancy_b_m: Vector3

    include_damping: bool = True
    include_restoring: bool = True
    include_added_mass: bool = False
    include_added_coriolis: bool = False

    @property
    def resolved_buoyancy_n(self) -> float:
        if self.buoyancy_n is not None:
            return float(self.buoyancy_n)
        if self.displaced_volume_m3 is None:
            raise ValueError("Expected 'buoyancy_n' or 'displaced_volume_m3'.")
        return float(
            self.displaced_volume_m3 * self.fluid_density_kg_m3 * self.gravity_m_s2
        )

    @property
    def resolved_displaced_volume_m3(self) -> float:
        if self.displaced_volume_m3 is not None:
            return float(self.displaced_volume_m3)
        return float(
            self.resolved_buoyancy_n / (self.fluid_density_kg_m3 * self.gravity_m_s2)
        )

    @model_validator(mode="after")
    def validate_resolved_pairs(self) -> HydroYamlModel:
        if self.buoyancy_n is not None and self.displaced_volume_m3 is not None:
            expected_buoyancy_n = (
                self.displaced_volume_m3 * self.fluid_density_kg_m3 * self.gravity_m_s2
            )
            if abs(self.buoyancy_n - expected_buoyancy_n) > _PAIR_TOLERANCE:
                raise ValueError(
                    "hydro.buoyancy_n and hydro.displaced_volume_m3 disagree."
                )

        if self.buoyancy_n is None and self.displaced_volume_m3 is None:
            raise ValueError(
                "Expected 'hydro.buoyancy_n' or 'hydro.displaced_volume_m3'."
            )

        return self


class ThrusterYamlModel(BaseModel):
    model_config = _MODEL_CONFIG

    model: str = Field(min_length=1)
    site_names: tuple[str, ...] = Field(min_length=1)


class DemoYamlModel(BaseModel):
    model_config = _MODEL_CONFIG

    surge_command: Vector8
    yaw_command: Vector8
    heave_command: Vector8
    coast_command: Vector8


class ViewerYamlModel(BaseModel):
    model_config = _MODEL_CONFIG

    distance: float
    elevation: float
    azimuth: float


class AUVYamlModel(BaseModel):
    model_config = _MODEL_CONFIG

    model_xml_path: str = Field(min_length=1)
    body_name: str = Field(min_length=1)
    hydro: HydroYamlModel
    thruster: ThrusterYamlModel
    body_wrench_limit: Vector6
    demo: DemoYamlModel
    viewer: ViewerYamlModel


@dataclass(frozen=True)
class AUVMjlabCfg:
    model_xml_path: str
    body_name: str
    thruster_model: str
    thruster_site_names: tuple[str, ...]
    linear_damping_matrix_6x6: Matrix6x6
    quadratic_damping_matrix_6x6: Matrix6x6
    current_velocity_w: Vector3
    current_velocity_b: Vector3 | None
    fluid_density_kg_m3: float
    gravity_m_s2: float
    displaced_volume_m3: float
    buoyancy_n: float
    center_of_buoyancy_b_m: Vector3
    center_of_gravity_b_m: Vector3
    added_mass_6x6: Matrix6x6
    include_damping: bool
    include_restoring: bool
    include_added_mass: bool
    include_added_coriolis: bool
    body_wrench_limit: Vector6
    surge_command: Vector8
    yaw_command: Vector8
    heave_command: Vector8
    coast_command: Vector8
    viewer_distance: float
    viewer_elevation: float
    viewer_azimuth: float


def _load_yaml_mapping(yaml_path: Path) -> dict[str, object]:
    with yaml_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML content must be a mapping in {yaml_path}")

    return raw


@lru_cache(maxsize=None)
def load_auv_cfg(yaml_path: Path) -> AUVMjlabCfg:
    raw = _load_yaml_mapping(yaml_path)

    parsed = AUVYamlModel.model_validate(raw)
    model_xml_path = _AUVRL_ROOT / parsed.model_xml_path
    center_of_gravity_b_m = _load_body_center_of_gravity(
        model_xml_path,
        parsed.body_name,
    )

    return AUVMjlabCfg(
        model_xml_path=parsed.model_xml_path,
        body_name=parsed.body_name,
        thruster_model=parsed.thruster.model,
        thruster_site_names=parsed.thruster.site_names,
        linear_damping_matrix_6x6=parsed.hydro.linear_damping_matrix,
        quadratic_damping_matrix_6x6=parsed.hydro.quadratic_damping_matrix,
        current_velocity_w=parsed.hydro.current_world_m_s,
        current_velocity_b=parsed.hydro.current_body_m_s,
        fluid_density_kg_m3=parsed.hydro.fluid_density_kg_m3,
        gravity_m_s2=parsed.hydro.gravity_m_s2,
        displaced_volume_m3=parsed.hydro.resolved_displaced_volume_m3,
        buoyancy_n=parsed.hydro.resolved_buoyancy_n,
        center_of_buoyancy_b_m=parsed.hydro.center_of_buoyancy_b_m,
        center_of_gravity_b_m=center_of_gravity_b_m,
        added_mass_6x6=parsed.hydro.added_mass_6x6,
        include_damping=parsed.hydro.include_damping,
        include_restoring=parsed.hydro.include_restoring,
        include_added_mass=parsed.hydro.include_added_mass,
        include_added_coriolis=parsed.hydro.include_added_coriolis,
        body_wrench_limit=parsed.body_wrench_limit,
        surge_command=parsed.demo.surge_command,
        yaw_command=parsed.demo.yaw_command,
        heave_command=parsed.demo.heave_command,
        coast_command=parsed.demo.coast_command,
        viewer_distance=parsed.viewer.distance,
        viewer_elevation=parsed.viewer.elevation,
        viewer_azimuth=parsed.viewer.azimuth,
    )
