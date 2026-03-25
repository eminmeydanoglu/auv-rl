"""Torch-first hydrodynamics utilities for the MJLab AUV stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch

import numpy as np
from mjlab.utils.lab_api.math import quat_apply_inverse

from auvrl.utils.type_aliases import _ZERO_3, _ZERO_6X6

VectorLike = Sequence[float] | np.ndarray | torch.Tensor
MatrixLike = Sequence[Sequence[float]] | np.ndarray | torch.Tensor


def _as_batch_vector(
    values: VectorLike,
    width: int,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.ndim == 1 and tensor.shape == (width,):
        return tensor.unsqueeze(0)
    if tensor.ndim == 2 and tensor.shape[1] == width:
        return tensor
    raise ValueError(
        f"{name} must have shape ({width},) or (batch, {width}), got "
        f"{tuple(tensor.shape)}"
    )


def _as_batch_matrix(
    values: MatrixLike,
    rows: int,
    cols: int,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(values, device=device, dtype=dtype)
    if tensor.shape == (rows, cols):
        return tensor.unsqueeze(0)
    if tensor.ndim == 3 and tensor.shape[1:] == (rows, cols):
        return tensor
    raise ValueError(
        f"{name} must have shape ({rows}, {cols}) or (batch, {rows}, {cols}), "
        f"got {tuple(tensor.shape)}"
    )


def _broadcast_batch(tensor: torch.Tensor, batch: int, name: str) -> torch.Tensor:
    if tensor.shape[0] == batch:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch, *tensor.shape[1:])
    raise ValueError(
        f"{name} has incompatible batch size {tensor.shape[0]} (expected {batch} or 1)"
    )


def _broadcast_scalar(
    value: float | torch.Tensor,
    batch: int,
    name: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    if tensor.ndim == 0:
        return tensor.expand(batch)
    if tensor.ndim == 1 and tensor.shape[0] == batch:
        return tensor
    if tensor.ndim == 1 and tensor.shape[0] == 1:
        return tensor.expand(batch)
    raise ValueError(
        f"{name} must be scalar or shape ({batch},), got {tuple(tensor.shape)}"
    )


def _batch_matvec(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.einsum("nij,nj->ni", matrix, vector)


def shift_wrench_origin_to_com(
    wrench_origin_b: VectorLike,
    center_of_gravity_b_m: VectorLike,
) -> torch.Tensor:
    """Shift a body-frame wrench from the body origin to the center of mass."""
    base = torch.as_tensor(wrench_origin_b)
    wrench = _as_batch_vector(
        wrench_origin_b,
        6,
        "wrench_origin_b",
        device=base.device,
        dtype=base.dtype if base.is_floating_point() else torch.float32,
    )
    cog = _as_batch_vector(
        center_of_gravity_b_m,
        3,
        "center_of_gravity_b_m",
        device=wrench.device,
        dtype=wrench.dtype,
    )
    cog = _broadcast_batch(cog, wrench.shape[0], "center_of_gravity_b_m")
    force = wrench[:, 0:3]
    torque_com = wrench[:, 3:6] - torch.cross(cog, force, dim=1)
    return torch.cat((force, torque_com), dim=1)


def shift_wrench_com_to_origin(
    wrench_com_b: VectorLike,
    center_of_gravity_b_m: VectorLike,
) -> torch.Tensor:
    """Shift a body-frame wrench from the center of mass to the body origin."""
    base = torch.as_tensor(wrench_com_b)
    wrench = _as_batch_vector(
        wrench_com_b,
        6,
        "wrench_com_b",
        device=base.device,
        dtype=base.dtype if base.is_floating_point() else torch.float32,
    )
    cog = _as_batch_vector(
        center_of_gravity_b_m,
        3,
        "center_of_gravity_b_m",
        device=wrench.device,
        dtype=wrench.dtype,
    )
    cog = _broadcast_batch(cog, wrench.shape[0], "center_of_gravity_b_m")
    force = wrench[:, 0:3]
    torque_origin = wrench[:, 3:6] + torch.cross(cog, force, dim=1)
    return torch.cat((force, torque_origin), dim=1)


@dataclass(frozen=True)
class AUVBodyState:
    """Vehicle state container used by the hydrodynamics model."""

    quat_wxyz: Optional[VectorLike] = None
    relative_twist_dot_body: Optional[VectorLike] = None

    lin_vel_body: Optional[VectorLike] = None
    ang_vel_body: Optional[VectorLike] = None
    lin_acc_body: Optional[VectorLike] = None
    ang_acc_body: Optional[VectorLike] = None

    lin_vel_world: Optional[VectorLike] = None
    ang_vel_world: Optional[VectorLike] = None
    lin_acc_world: Optional[VectorLike] = None
    ang_acc_world: Optional[VectorLike] = None


@dataclass(frozen=True)
class HydroConfig:
    """Hydrodynamics configuration in ``[u,v,w,p,q,r]`` ordering."""

    center_of_gravity_b_m: VectorLike | None = None
    center_of_buoyancy_b_m: VectorLike = _ZERO_3

    fluid_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.81
    weight_n: float | None = None
    buoyancy_n: float | None = None
    displaced_volume_m3: float | None = None

    current_world_m_s: VectorLike | None = _ZERO_3
    current_body_m_s: VectorLike | None = None

    linear_damping_matrix_6x6: MatrixLike = _ZERO_6X6
    quadratic_damping_matrix_6x6: MatrixLike = _ZERO_6X6
    added_mass_6x6: MatrixLike = _ZERO_6X6

    include_restoring: bool = True
    include_damping: bool = True
    include_added_mass: bool = False
    include_added_coriolis: bool = False

    @property
    def resolved_buoyancy_n(self) -> float:
        if self.buoyancy_n is not None:
            return float(self.buoyancy_n)
        if self.displaced_volume_m3 is None:
            return 0.0
        return (
            float(self.fluid_density_kg_m3)
            * float(self.gravity_m_s2)
            * float(self.displaced_volume_m3)
        )


def added_mass_coriolis_wrench(
    added_mass_6x6: MatrixLike,
    relative_twist_b: VectorLike,
) -> torch.Tensor:
    """Return the hydrodynamic added-mass wrench ``-C_A(nu_r) nu_r``."""
    base = torch.as_tensor(relative_twist_b)
    twist = _as_batch_vector(
        relative_twist_b,
        6,
        "relative_twist_b",
        device=base.device,
        dtype=base.dtype if base.is_floating_point() else torch.float32,
    )
    matrix = _as_batch_matrix(
        added_mass_6x6,
        6,
        6,
        "added_mass_6x6",
        device=twist.device,
        dtype=twist.dtype,
    )
    matrix = _broadcast_batch(matrix, twist.shape[0], "added_mass_6x6")

    ma11 = matrix[:, 0:3, 0:3]
    ma12 = matrix[:, 0:3, 3:6]
    ma21 = matrix[:, 3:6, 0:3]
    ma22 = matrix[:, 3:6, 3:6]

    nu1 = twist[:, 0:3]
    nu2 = twist[:, 3:6]

    a = _batch_matvec(ma11, nu1) + _batch_matvec(ma12, nu2)
    b = _batch_matvec(ma21, nu1) + _batch_matvec(ma22, nu2)

    force = torch.cross(a, nu2, dim=1)
    torque = torch.cross(a, nu1, dim=1) + torch.cross(b, nu2, dim=1)
    return torch.cat((force, torque), dim=1)


def compute_hydrodynamic_wrench(
    *,
    quat_wxyz: VectorLike | None,
    relative_twist_b: VectorLike,
    linear_damping_matrix_6x6: MatrixLike,
    quadratic_damping_matrix_6x6: MatrixLike,
    added_mass_6x6: MatrixLike,
    center_of_buoyancy_b_m: VectorLike,
    buoyancy_n: float | torch.Tensor,
    center_of_gravity_b_m: VectorLike | None = None,
    weight_n: float | torch.Tensor | None = None,
    relative_twist_dot_b: VectorLike | None = None,
    include_damping: bool = True,
    include_restoring: bool = True,
    include_added_mass: bool = False,
    include_added_coriolis: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute the full body-origin hydrodynamic wrench.

    The quaternion is interpreted as body-to-world and outputs are always batched.
    """
    base = torch.as_tensor(relative_twist_b)
    device = base.device
    dtype = base.dtype if base.is_floating_point() else torch.float32

    twist = _as_batch_vector(
        relative_twist_b,
        6,
        "relative_twist_b",
        device=device,
        dtype=dtype,
    )
    batch = twist.shape[0]

    linear_damping = _as_batch_matrix(
        linear_damping_matrix_6x6,
        6,
        6,
        "linear_damping_matrix_6x6",
        device=device,
        dtype=dtype,
    )
    quadratic_damping = _as_batch_matrix(
        quadratic_damping_matrix_6x6,
        6,
        6,
        "quadratic_damping_matrix_6x6",
        device=device,
        dtype=dtype,
    )
    added_mass = _as_batch_matrix(
        added_mass_6x6,
        6,
        6,
        "added_mass_6x6",
        device=device,
        dtype=dtype,
    )

    linear_damping = _broadcast_batch(
        linear_damping,
        batch,
        "linear_damping_matrix_6x6",
    )
    quadratic_damping = _broadcast_batch(
        quadratic_damping,
        batch,
        "quadratic_damping_matrix_6x6",
    )
    added_mass = _broadcast_batch(added_mass, batch, "added_mass_6x6")

    tau_total = torch.zeros((batch, 6), device=device, dtype=dtype)

    if include_damping:
        tau_damping = -(
            _batch_matvec(linear_damping, twist)
            + _batch_matvec(quadratic_damping, twist.abs() * twist)
        )
        tau_total += tau_damping
    else:
        tau_damping = torch.zeros_like(tau_total)

    if include_restoring:
        if quat_wxyz is None:
            raise ValueError(
                "Restoring forces require quat_wxyz (body-to-world quaternion)."
            )
        quat = _as_batch_vector(
            quat_wxyz,
            4,
            "quat_wxyz",
            device=device,
            dtype=dtype,
        )
        quat = _broadcast_batch(quat, batch, "quat_wxyz")

        cob = _as_batch_vector(
            center_of_buoyancy_b_m,
            3,
            "center_of_buoyancy_b_m",
            device=device,
            dtype=dtype,
        )
        cob = _broadcast_batch(cob, batch, "center_of_buoyancy_b_m")
        buoyancy = _broadcast_scalar(
            buoyancy_n,
            batch,
            "buoyancy_n",
            device=device,
            dtype=dtype,
        )

        force_buoy_w = torch.zeros((batch, 3), device=device, dtype=dtype)
        force_buoy_w[:, 2] = buoyancy
        force_buoy_b = quat_apply_inverse(quat, force_buoy_w)
        torque_buoy_b = torch.cross(cob, force_buoy_b, dim=1)

        if center_of_gravity_b_m is None or weight_n is None:
            raise ValueError(
                "Restoring forces require weight_n and center_of_gravity_b_m."
            )

        cog = _as_batch_vector(
            center_of_gravity_b_m,
            3,
            "center_of_gravity_b_m",
            device=device,
            dtype=dtype,
        )
        cog = _broadcast_batch(cog, batch, "center_of_gravity_b_m")
        weight = _broadcast_scalar(
            weight_n,
            batch,
            "weight_n",
            device=device,
            dtype=dtype,
        )

        force_weight_w = torch.zeros((batch, 3), device=device, dtype=dtype)
        force_weight_w[:, 2] = -weight
        force_weight_b = quat_apply_inverse(quat, force_weight_w)
        torque_weight_b = torch.cross(cog, force_weight_b, dim=1)

        tau_restoring = torch.cat(
            (
                force_buoy_b + force_weight_b,
                torque_buoy_b + torque_weight_b,
            ),
            dim=1,
        )

        tau_total += tau_restoring
    else:
        tau_restoring = torch.zeros_like(tau_total)

    if include_added_mass:
        if relative_twist_dot_b is None:
            raise ValueError("Added-mass acceleration requires relative_twist_dot_b.")
        twist_dot = _as_batch_vector(
            relative_twist_dot_b,
            6,
            "relative_twist_dot_b",
            device=device,
            dtype=dtype,
        )
        twist_dot = _broadcast_batch(twist_dot, batch, "relative_twist_dot_b")
        tau_added_mass = -_batch_matvec(added_mass, twist_dot)
        tau_total += tau_added_mass
    else:
        tau_added_mass = torch.zeros_like(tau_total)

    if include_added_coriolis:
        tau_added_coriolis = added_mass_coriolis_wrench(added_mass, twist)
        tau_total += tau_added_coriolis
    else:
        tau_added_coriolis = torch.zeros_like(tau_total)

    return {
        "tau_total_body": tau_total,
        "tau_damping_body": tau_damping,
        "tau_restoring_body": tau_restoring,
        "tau_added_mass_body": tau_added_mass,
        "tau_added_coriolis_body": tau_added_coriolis,
    }


class HydrodynamicsModel:
    """Compute hydrodynamic wrench from vehicle state."""

    def __init__(
        self,
        config: Optional[HydroConfig] = None,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config or HydroConfig()
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.dtype = dtype

        self._linear_damping = torch.as_tensor(
            self.config.linear_damping_matrix_6x6,
            device=self.device,
            dtype=self.dtype,
        )
        self._quadratic_damping = torch.as_tensor(
            self.config.quadratic_damping_matrix_6x6,
            device=self.device,
            dtype=self.dtype,
        )
        self._added_mass = torch.as_tensor(
            self.config.added_mass_6x6,
            device=self.device,
            dtype=self.dtype,
        )
        self._current_world = (
            None
            if self.config.current_world_m_s is None
            else torch.as_tensor(
                self.config.current_world_m_s,
                device=self.device,
                dtype=self.dtype,
            )
        )
        self._current_body = (
            None
            if self.config.current_body_m_s is None
            else torch.as_tensor(
                self.config.current_body_m_s,
                device=self.device,
                dtype=self.dtype,
            )
        )

    def _extract_body_motion(
        self,
        state: AUVBodyState,
        expected_batch: Optional[int] = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        quat_wxyz: Optional[torch.Tensor] = None
        batch = expected_batch

        if state.quat_wxyz is not None:
            quat_wxyz = _as_batch_vector(
                state.quat_wxyz,
                4,
                "quat_wxyz",
                device=self.device,
                dtype=self.dtype,
            )
            if batch is None:
                batch = quat_wxyz.shape[0]

        if state.lin_vel_body is not None:
            vel_body = _as_batch_vector(
                state.lin_vel_body,
                3,
                "lin_vel_body",
                device=self.device,
                dtype=self.dtype,
            )
        elif state.lin_vel_world is not None and quat_wxyz is not None:
            vel_world = _as_batch_vector(
                state.lin_vel_world,
                3,
                "lin_vel_world",
                device=self.device,
                dtype=self.dtype,
            )
            batch = vel_world.shape[0] if batch is None else batch
            vel_world = _broadcast_batch(vel_world, batch, "lin_vel_world")
            quat_wxyz = _broadcast_batch(quat_wxyz, batch, "quat_wxyz")
            vel_body = quat_apply_inverse(quat_wxyz, vel_world)
        else:
            raise ValueError("Provide lin_vel_body, or (lin_vel_world + quat_wxyz).")

        batch = vel_body.shape[0] if batch is None else batch
        vel_body = _broadcast_batch(vel_body, batch, "lin_vel_body")

        if state.ang_vel_body is not None:
            omega_body = _as_batch_vector(
                state.ang_vel_body,
                3,
                "ang_vel_body",
                device=self.device,
                dtype=self.dtype,
            )
        elif state.ang_vel_world is not None and quat_wxyz is not None:
            omega_world = _as_batch_vector(
                state.ang_vel_world,
                3,
                "ang_vel_world",
                device=self.device,
                dtype=self.dtype,
            )
            omega_world = _broadcast_batch(omega_world, batch, "ang_vel_world")
            quat_wxyz = _broadcast_batch(quat_wxyz, batch, "quat_wxyz")
            omega_body = quat_apply_inverse(quat_wxyz, omega_world)
        else:
            raise ValueError("Provide ang_vel_body, or (ang_vel_world + quat_wxyz).")

        omega_body = _broadcast_batch(omega_body, batch, "ang_vel_body")

        lin_acc_body: Optional[torch.Tensor]
        ang_acc_body: Optional[torch.Tensor]

        if state.lin_acc_body is not None:
            lin_acc_body = _as_batch_vector(
                state.lin_acc_body,
                3,
                "lin_acc_body",
                device=self.device,
                dtype=self.dtype,
            )
            lin_acc_body = _broadcast_batch(lin_acc_body, batch, "lin_acc_body")
        elif state.lin_acc_world is not None and quat_wxyz is not None:
            lin_acc_world = _as_batch_vector(
                state.lin_acc_world,
                3,
                "lin_acc_world",
                device=self.device,
                dtype=self.dtype,
            )
            lin_acc_world = _broadcast_batch(lin_acc_world, batch, "lin_acc_world")
            quat_wxyz = _broadcast_batch(quat_wxyz, batch, "quat_wxyz")
            lin_acc_body = quat_apply_inverse(quat_wxyz, lin_acc_world)
        else:
            lin_acc_body = None

        if state.ang_acc_body is not None:
            ang_acc_body = _as_batch_vector(
                state.ang_acc_body,
                3,
                "ang_acc_body",
                device=self.device,
                dtype=self.dtype,
            )
            ang_acc_body = _broadcast_batch(ang_acc_body, batch, "ang_acc_body")
        elif state.ang_acc_world is not None and quat_wxyz is not None:
            ang_acc_world = _as_batch_vector(
                state.ang_acc_world,
                3,
                "ang_acc_world",
                device=self.device,
                dtype=self.dtype,
            )
            ang_acc_world = _broadcast_batch(ang_acc_world, batch, "ang_acc_world")
            quat_wxyz = _broadcast_batch(quat_wxyz, batch, "quat_wxyz")
            ang_acc_body = quat_apply_inverse(quat_wxyz, ang_acc_world)
        else:
            ang_acc_body = None

        return vel_body, omega_body, lin_acc_body, ang_acc_body, quat_wxyz

    def _current_body_kinematics(
        self,
        batch: int,
        quat_wxyz: Optional[torch.Tensor],
        omega_body: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._current_body is not None:
            current_body = _as_batch_vector(
                self._current_body,
                3,
                "current_body_m_s",
                device=self.device,
                dtype=self.dtype,
            )
            current_body = _broadcast_batch(current_body, batch, "current_body_m_s")
            # Body-frame current is treated as body-fixed, so its body derivative is zero.
            return current_body, torch.zeros_like(current_body)

        if self._current_world is None:
            zeros = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
            return zeros, zeros

        current_world = _as_batch_vector(
            self._current_world,
            3,
            "current_world_m_s",
            device=self.device,
            dtype=self.dtype,
        )
        current_world = _broadcast_batch(current_world, batch, "current_world_m_s")

        if quat_wxyz is None:
            if bool(torch.allclose(current_world, torch.zeros_like(current_world))):
                zeros = torch.zeros((batch, 3), device=self.device, dtype=self.dtype)
                return zeros, zeros
            raise ValueError(
                "current_world_m_s is non-zero but quat_wxyz is missing. Provide "
                "quat_wxyz or set current_body_m_s."
            )

        quat_wxyz = _broadcast_batch(quat_wxyz, batch, "quat_wxyz")
        current_body = quat_apply_inverse(quat_wxyz, current_world)
        current_body_dot = -torch.cross(omega_body, current_body, dim=1)
        return current_body, current_body_dot

    def compute_wrench(
        self,
        state: AUVBodyState,
        expected_batch: Optional[int] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the hydrodynamic wrench about the body-frame origin."""
        vel_b, omega_b, lin_acc_b, ang_acc_b, quat_wxyz = self._extract_body_motion(
            state,
            expected_batch=expected_batch,
        )
        batch = vel_b.shape[0]

        current_body, current_body_dot = self._current_body_kinematics(
            batch,
            quat_wxyz,
            omega_b,
        )
        nu = torch.cat((vel_b, omega_b), dim=1)
        nu_r = torch.cat((vel_b - current_body, omega_b), dim=1)

        nu_dot_r: Optional[torch.Tensor] = None
        if state.relative_twist_dot_body is not None:
            nu_dot_r = _as_batch_vector(
                state.relative_twist_dot_body,
                6,
                "relative_twist_dot_body",
                device=self.device,
                dtype=self.dtype,
            )
            nu_dot_r = _broadcast_batch(nu_dot_r, batch, "relative_twist_dot_body")
        elif lin_acc_b is not None and ang_acc_b is not None:
            nu_dot_r = torch.cat((lin_acc_b - current_body_dot, ang_acc_b), dim=1)

        results = compute_hydrodynamic_wrench(
            quat_wxyz=state.quat_wxyz,
            relative_twist_b=nu_r,
            relative_twist_dot_b=nu_dot_r,
            linear_damping_matrix_6x6=self._linear_damping,
            quadratic_damping_matrix_6x6=self._quadratic_damping,
            added_mass_6x6=self._added_mass,
            center_of_buoyancy_b_m=self.config.center_of_buoyancy_b_m,
            center_of_gravity_b_m=self.config.center_of_gravity_b_m,
            buoyancy_n=self.config.resolved_buoyancy_n,
            weight_n=self.config.weight_n,
            include_damping=self.config.include_damping,
            include_restoring=self.config.include_restoring,
            include_added_mass=self.config.include_added_mass,
            include_added_coriolis=self.config.include_added_coriolis,
        )

        diagnostics = {
            "nu_body": nu,
            "nu_r_body": nu_r,
            "nu_dot_r_body": (
                nu_dot_r
                if nu_dot_r is not None
                else torch.zeros_like(nu_r, device=self.device, dtype=self.dtype)
            ),
            "current_body": current_body,
            **results,
        }
        return results["tau_total_body"], diagnostics
