"""Thruster MJLab config models and loaders."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


_MODEL_CONFIG = ConfigDict(extra="forbid", frozen=True)

THRUSTER_CFG_DIR = Path(__file__).resolve().parents[1] / "asset_zoo" / "thrusters"

Polynomial6 = tuple[float, float, float, float, float, float]
SupplyVoltage = float | tuple[float, ...]


class ThrusterModelCfg(BaseModel):
    model_config = _MODEL_CONFIG

    command_limit: float = Field(gt=0.0)
    tau_s: float = Field(gt=0.0)
    force_deadzone_n: float = Field(ge=0.0)
    min_thrust_n: float
    max_thrust_n: float
    supply_voltage: SupplyVoltage
    pwm_min_us: float
    pwm_max_us: float
    pwm_neutral_us: float
    force_to_pwm_coeffs_forward: Polynomial6
    force_to_pwm_coeffs_reverse: Polynomial6
    newton_per_kgf: float = Field(gt=0.0)

    @field_validator("supply_voltage")
    @classmethod
    def validate_supply_voltage(cls, value: SupplyVoltage) -> SupplyVoltage:
        if isinstance(value, tuple):
            if len(value) == 0:
                raise ValueError("supply_voltage must not be empty.")
            if any(float(entry) <= 0.0 for entry in value):
                raise ValueError("supply_voltage entries must all be positive.")
            return tuple(float(entry) for entry in value)

        scalar = float(value)
        if scalar <= 0.0:
            raise ValueError("supply_voltage must be positive.")
        return scalar

    @model_validator(mode="after")
    def validate_ranges(self) -> ThrusterModelCfg:
        if self.max_thrust_n < self.min_thrust_n:
            raise ValueError("max_thrust_n must be >= min_thrust_n.")

        pwm_min = min(self.pwm_min_us, self.pwm_max_us)
        pwm_max = max(self.pwm_min_us, self.pwm_max_us)
        if not pwm_min <= self.pwm_neutral_us <= pwm_max:
            raise ValueError("pwm_neutral_us must lie within [pwm_min_us, pwm_max_us].")

        return self


def _load_yaml_mapping(yaml_path: Path) -> dict[str, object]:
    with yaml_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML content must be a mapping in {yaml_path}")

    return raw


@lru_cache(maxsize=None)
def load_thruster_cfg(yaml_path: Path) -> ThrusterModelCfg:
    yaml_path = yaml_path.resolve()
    raw = _load_yaml_mapping(yaml_path)
    return ThrusterModelCfg.model_validate(raw)
