"""MJLab site thruster actuator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence, cast

import mujoco
import mujoco_warp as mjwarp
import torch

from mjlab.actuator.actuator import (
    Actuator,
    ActuatorCfg,
    ActuatorCmd,
    TransmissionType,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity

    from auvrl.config.thruster_cfg import ThrusterModelCfg


THRUSTER_LOCAL_AXIS = (0.0, 0.0, -1.0)


@dataclass(kw_only=True)
class ThrusterActuatorCfg(ActuatorCfg):
    """Force-target thruster actuator using a voltage-aware calibration model.

    Positive thrust acts along the MuJoCo site local ``-Z`` axis.
    Commands are interpreted as per-thruster force targets in newtons.
    """

    transmission_type: TransmissionType = TransmissionType.SITE

    tau_s: float
    command_limit: float
    force_deadzone_n: float
    min_thrust_n: float
    max_thrust_n: float
    supply_voltage: float | Sequence[float]
    pwm_min_us: float
    pwm_max_us: float
    pwm_neutral_us: float
    force_to_pwm_coeffs_forward: tuple[float, float, float, float, float, float]
    force_to_pwm_coeffs_reverse: tuple[float, float, float, float, float, float]
    newton_per_kgf: float

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.transmission_type != TransmissionType.SITE:
            raise ValueError("ThrusterActuatorCfg requires transmission_type=SITE.")
        if self.tau_s <= 0.0:
            raise ValueError("tau_s must be positive.")
        if self.command_limit <= 0.0:
            raise ValueError("command_limit must be positive.")
        if self.force_deadzone_n < 0.0:
            raise ValueError("force_deadzone_n must be non-negative.")
        if self.max_thrust_n < self.min_thrust_n:
            raise ValueError("max_thrust_n must be >= min_thrust_n.")
        if self.newton_per_kgf <= 0.0:
            raise ValueError("newton_per_kgf must be positive.")
        if len(self.force_to_pwm_coeffs_forward) != 6:
            raise ValueError("force_to_pwm_coeffs_forward must have 6 values.")
        if len(self.force_to_pwm_coeffs_reverse) != 6:
            raise ValueError("force_to_pwm_coeffs_reverse must have 6 values.")

        if isinstance(self.supply_voltage, (int, float)):
            if float(self.supply_voltage) <= 0.0:
                raise ValueError("supply_voltage must be positive.")
        else:
            supply_voltage = cast(Sequence[float], self.supply_voltage)
            if len(supply_voltage) == 0:
                raise ValueError("supply_voltage must not be empty.")
            if any(float(value) <= 0.0 for value in supply_voltage):
                raise ValueError("supply_voltage entries must all be positive.")

        pwm_min = min(self.pwm_min_us, self.pwm_max_us)
        pwm_max = max(self.pwm_min_us, self.pwm_max_us)
        if not pwm_min <= self.pwm_neutral_us <= pwm_max:
            raise ValueError("pwm_neutral_us must lie within [pwm_min_us, pwm_max_us].")

    def build(
        self,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> ThrusterActuator:
        return ThrusterActuator(self, entity, target_ids, target_names)


def make_thruster_actuator_cfg(
    *,
    target_names_expr: tuple[str, ...],
    thruster_cfg: ThrusterModelCfg,
) -> ThrusterActuatorCfg:
    """Build a `ThrusterActuatorCfg` from a shared thruster model config."""

    return ThrusterActuatorCfg(
        target_names_expr=target_names_expr,
        tau_s=thruster_cfg.tau_s,
        command_limit=thruster_cfg.command_limit,
        force_deadzone_n=thruster_cfg.force_deadzone_n,
        min_thrust_n=thruster_cfg.min_thrust_n,
        max_thrust_n=thruster_cfg.max_thrust_n,
        supply_voltage=thruster_cfg.supply_voltage,
        pwm_min_us=thruster_cfg.pwm_min_us,
        pwm_max_us=thruster_cfg.pwm_max_us,
        pwm_neutral_us=thruster_cfg.pwm_neutral_us,
        force_to_pwm_coeffs_forward=thruster_cfg.force_to_pwm_coeffs_forward,
        force_to_pwm_coeffs_reverse=thruster_cfg.force_to_pwm_coeffs_reverse,
        newton_per_kgf=thruster_cfg.newton_per_kgf,
    )


class ThrusterActuator(Actuator[ThrusterActuatorCfg]):
    """Custom site actuator implementing force-target thruster dynamics."""

    def __init__(
        self,
        cfg: ThrusterActuatorCfg,
        entity: Entity,
        target_ids: list[int],
        target_names: list[str],
    ) -> None:
        super().__init__(cfg, entity, target_ids, target_names)
        self._dt_s: float = 1.0 / 500.0
        self._thrust_state: torch.Tensor | None = None
        self._supply_voltage: torch.Tensor | None = None
        self._force_pwm_forward_coeffs: torch.Tensor | None = None
        self._force_pwm_reverse_coeffs: torch.Tensor | None = None

    def edit_spec(self, spec: Any, target_names: list[str]) -> None:
        mj_module: Any = mujoco
        effort_limit = max(abs(self.cfg.min_thrust_n), abs(self.cfg.max_thrust_n), 1e-6)
        for target_name in target_names:
            actuator = spec.add_actuator(name=target_name, target=target_name)
            actuator.trntype = mj_module.mjtTrn.mjTRN_SITE
            actuator.dyntype = mj_module.mjtDyn.mjDYN_NONE
            actuator.gaintype = mj_module.mjtGain.mjGAIN_FIXED
            actuator.biastype = mj_module.mjtBias.mjBIAS_NONE
            actuator.gear[:] = (
                THRUSTER_LOCAL_AXIS[0],
                THRUSTER_LOCAL_AXIS[1],
                THRUSTER_LOCAL_AXIS[2],
                0.0,
                0.0,
                0.0,
            )
            actuator.forcelimited = True
            actuator.forcerange[:] = (-effort_limit, effort_limit)
            actuator.ctrllimited = True
            actuator.ctrlrange[:] = (-effort_limit, effort_limit)
            self._mjs_actuators.append(actuator)

    def initialize(
        self,
        mj_model: Any,
        model: mjwarp.Model,
        data: mjwarp.Data,
        device: str,
    ) -> None:
        super().initialize(mj_model, model, data, device)
        self._dt_s = float(mj_model.opt.timestep)
        self._thrust_state = torch.zeros(
            (data.nworld, len(self._target_ids_list)),
            device=device,
            dtype=torch.float,
        )
        self._supply_voltage = self._expand_voltage(
            self.cfg.supply_voltage,
            num_envs=data.nworld,
            device=device,
        )
        self._force_pwm_forward_coeffs = torch.tensor(
            self.cfg.force_to_pwm_coeffs_forward,
            device=device,
            dtype=torch.float,
        )
        self._force_pwm_reverse_coeffs = torch.tensor(
            self.cfg.force_to_pwm_coeffs_reverse,
            device=device,
            dtype=torch.float,
        )

    @property
    def supply_voltage(self) -> torch.Tensor:
        if self._supply_voltage is None:
            raise RuntimeError("Actuator must be initialized before reading voltage.")
        return self._supply_voltage

    @property
    def thrust_state(self) -> torch.Tensor:
        if self._thrust_state is None:
            raise RuntimeError("Actuator must be initialized before reading thrust.")
        return self._thrust_state

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        thrust_target = self._force_target_thrust_target(cmd.effort_target)
        if self._thrust_state is None:
            raise RuntimeError("Thruster state was not initialized.")

        alpha = math.exp(-self._dt_s / max(self.cfg.tau_s, 1e-6))
        self._thrust_state = (
            alpha * self._thrust_state + (1.0 - alpha) * thrust_target
        ).clamp(self.cfg.min_thrust_n, self.cfg.max_thrust_n)
        return self._thrust_state

    def update(self, dt: float) -> None:
        if dt > 0.0:
            self._dt_s = float(dt)

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if self._thrust_state is None:
            return
        if env_ids is None:
            env_ids = slice(None)
        self._thrust_state[env_ids] = 0.0

    def set_supply_voltage(
        self,
        voltage: float | Sequence[float] | torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if self._supply_voltage is None:
            raise RuntimeError("Actuator must be initialized before setting voltage.")
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            num_envs = self._supply_voltage[env_ids].shape[0]
        else:
            num_envs = int(env_ids.numel())

        updated = self._expand_voltage(
            voltage,
            num_envs=num_envs,
            device=str(self._supply_voltage.device),
        )
        self._supply_voltage[env_ids] = updated

    def _force_target_thrust_target(self, command: torch.Tensor) -> torch.Tensor:
        if (
            self._supply_voltage is None
            or self._force_pwm_forward_coeffs is None
            or self._force_pwm_reverse_coeffs is None
        ):
            raise RuntimeError("Force-target model tensors were not initialized.")

        limit = max(self.cfg.command_limit, 1e-6)
        force_cmd = command.clamp(-limit, limit)
        deadzone = abs(self.cfg.force_deadzone_n)
        if deadzone > 0.0:
            force_cmd = torch.where(
                force_cmd.abs() < deadzone,
                torch.zeros_like(force_cmd),
                force_cmd,
            )

        coeffs = torch.where(
            (force_cmd >= 0.0).unsqueeze(-1),
            self._force_pwm_forward_coeffs.view(1, 1, 6),
            self._force_pwm_reverse_coeffs.view(1, 1, 6),
        )

        newton_per_kgf = max(self.cfg.newton_per_kgf, 1e-12)
        force_kgf = force_cmd / newton_per_kgf

        a, b, c, d, e, f = coeffs.unbind(dim=-1)
        voltage = self._supply_voltage
        pwm_req = (
            a * force_kgf * force_kgf
            + b * force_kgf * voltage
            + c * voltage * voltage
            + d * force_kgf
            + e * voltage
            + f
        )

        pwm_min = min(self.cfg.pwm_min_us, self.cfg.pwm_max_us)
        pwm_max = max(self.cfg.pwm_min_us, self.cfg.pwm_max_us)
        pwm_sat = pwm_req.clamp(pwm_min, pwm_max)
        neutral_mask = (pwm_sat - self.cfg.pwm_neutral_us).abs() < 1e-9

        achievable_force = self._pwm_to_force_newton(
            pwm_sat,
            voltage,
            coeffs,
            hint_force_newton=force_cmd,
        )
        achievable_force = torch.where(
            neutral_mask,
            torch.zeros_like(achievable_force),
            achievable_force,
        )
        return achievable_force.clamp(self.cfg.min_thrust_n, self.cfg.max_thrust_n)

    def _pwm_to_force_newton(
        self,
        pwm_us: torch.Tensor,
        voltage: torch.Tensor,
        coeffs: torch.Tensor,
        hint_force_newton: torch.Tensor,
    ) -> torch.Tensor:
        a, b, c, d, e, f = coeffs.unbind(dim=-1)
        beta = b * voltage + d
        gamma = c * voltage * voltage + e * voltage + f - pwm_us

        eps = 1e-12
        newton_per_kgf = max(self.cfg.newton_per_kgf, eps)
        hint_kgf = hint_force_newton / newton_per_kgf
        roots = hint_kgf.clone()

        linear_mask = a.abs() < eps
        beta_nonzero = beta.abs() >= eps
        linear_root = torch.where(
            beta_nonzero,
            -gamma / torch.where(beta_nonzero, beta, torch.ones_like(beta)),
            torch.zeros_like(beta),
        )
        roots = torch.where(linear_mask, linear_root, roots)

        quad_mask = ~linear_mask
        safe_a = torch.where(quad_mask, a, torch.ones_like(a))
        discriminant = beta * beta - 4.0 * a * gamma
        has_real_roots = discriminant >= -1e-9
        discriminant = discriminant.clamp_min(0.0)
        sqrt_discriminant = torch.sqrt(discriminant)

        root_a = (-beta + sqrt_discriminant) / (2.0 * safe_a)
        root_b = (-beta - sqrt_discriminant) / (2.0 * safe_a)
        selected_root = self._pick_root(root_a, root_b, hint_kgf)

        use_selected = quad_mask & has_real_roots
        roots = torch.where(use_selected, selected_root, roots)
        roots = torch.where(quad_mask & (~has_real_roots), hint_kgf, roots)
        return roots * newton_per_kgf

    @staticmethod
    def _pick_root(
        root_a: torch.Tensor,
        root_b: torch.Tensor,
        hint: torch.Tensor,
    ) -> torch.Tensor:
        preferred_sign = torch.sign(hint)
        sign_enabled = preferred_sign.abs() > 1e-9

        finite_a = torch.isfinite(root_a)
        finite_b = torch.isfinite(root_b)

        sign_ok_a = (~sign_enabled) | (root_a * preferred_sign >= -1e-9)
        sign_ok_b = (~sign_enabled) | (root_b * preferred_sign >= -1e-9)

        valid_a = finite_a & sign_ok_a
        valid_b = finite_b & sign_ok_b

        dist_a = (root_a - hint).abs()
        dist_b = (root_b - hint).abs()

        selected = hint.clone()
        both_valid = valid_a & valid_b
        pick_a = both_valid & (dist_a <= dist_b)
        pick_b = both_valid & (~pick_a)
        selected = torch.where(pick_a, root_a, selected)
        selected = torch.where(pick_b, root_b, selected)
        selected = torch.where(valid_a & (~valid_b), root_a, selected)
        selected = torch.where(valid_b & (~valid_a), root_b, selected)

        none_valid = (~valid_a) & (~valid_b)
        any_finite = finite_a | finite_b
        pick_a_fallback = none_valid & any_finite & ((~finite_b) | (dist_a <= dist_b))
        pick_b_fallback = none_valid & finite_b & (~pick_a_fallback)
        selected = torch.where(pick_a_fallback, root_a, selected)
        selected = torch.where(pick_b_fallback, root_b, selected)
        return selected

    def _expand_voltage(
        self,
        value: float | Sequence[float] | torch.Tensor,
        *,
        num_envs: int,
        device: str,
    ) -> torch.Tensor:
        num_thrusters = len(self._target_ids_list)
        tensor = torch.as_tensor(value, dtype=torch.float, device=device)
        if tensor.ndim == 0:
            return torch.full(
                (num_envs, num_thrusters),
                float(tensor.item()),
                device=device,
            )
        if tensor.ndim == 1 and tensor.shape[0] == num_thrusters:
            return tensor.unsqueeze(0).expand(num_envs, -1).clone()
        if tensor.ndim == 2 and tensor.shape == (num_envs, num_thrusters):
            return tensor.clone()
        raise ValueError(
            "supply_voltage must be scalar, shape (num_thrusters,), or shape "
            f"({num_envs}, {num_thrusters}). Received {tuple(tensor.shape)}."
        )
