"""Hydrodynamics action term for MJLab AUV tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse

from auvrl.utils.type_aliases import Matrix6x6, Vector3, _ZERO_6X6

from .hydrodynamics import (
    compute_hydrodynamic_wrench,
    shift_wrench_origin_to_com,
)

if TYPE_CHECKING:
    from mjlab.entity import Entity
    from mjlab.envs import ManagerBasedRlEnv

    from auvrl.config.auv_cfg import AUVMjlabCfg


@dataclass(kw_only=True)
class UnderwaterHydroActionCfg(ActionTermCfg):
    """Configuration for per-physics-step underwater hydrodynamics."""

    body_name: str = "base_link"

    linear_damping_matrix_6x6: Matrix6x6 = _ZERO_6X6
    quadratic_damping_matrix_6x6: Matrix6x6 = _ZERO_6X6

    current_velocity_w: Vector3 = (0.0, 0.0, 0.0)
    current_velocity_b: Vector3 | None = None

    fluid_density_kg_m3: float = 1025.0
    gravity_m_s2: float = 9.81
    displaced_volume_m3: float = 0.0
    buoyancy_n: float | None = None

    center_of_buoyancy_b_m: Vector3 = (0.0, 0.0, 0.0)

    added_mass_6x6: Matrix6x6 = _ZERO_6X6

    include_damping: bool = True
    include_restoring: bool = True
    include_added_mass: bool = False
    include_added_coriolis: bool = False

    def __post_init__(self) -> None:
        vector_fields = {
            "current_velocity_w": self.current_velocity_w,
            "center_of_buoyancy_b_m": self.center_of_buoyancy_b_m,
        }
        if self.current_velocity_b is not None:
            vector_fields["current_velocity_b"] = self.current_velocity_b
        for name, values in vector_fields.items():
            if len(values) != 3:
                raise ValueError(f"{name} must contain exactly 3 values.")

        if len(self.added_mass_6x6) != 6 or any(
            len(row) != 6 for row in self.added_mass_6x6
        ):
            raise ValueError("added_mass_6x6 must have shape (6, 6).")

        for name, matrix in {
            "linear_damping_matrix_6x6": self.linear_damping_matrix_6x6,
            "quadratic_damping_matrix_6x6": self.quadratic_damping_matrix_6x6,
        }.items():
            if len(matrix) != 6 or any(len(row) != 6 for row in matrix):
                raise ValueError(f"{name} must have shape (6, 6).")

        if self.fluid_density_kg_m3 < 0.0:
            raise ValueError("fluid_density_kg_m3 must be non-negative.")
        if self.gravity_m_s2 < 0.0:
            raise ValueError("gravity_m_s2 must be non-negative.")
        if self.displaced_volume_m3 < 0.0:
            raise ValueError("displaced_volume_m3 must be non-negative.")
        if self.buoyancy_n is not None and self.buoyancy_n < 0.0:
            raise ValueError("buoyancy_n must be non-negative.")

    def build(self, env: ManagerBasedRlEnv) -> UnderwaterHydroAction:
        return UnderwaterHydroAction(self, env)


def make_underwater_hydro_action_cfg(
    *,
    auv_cfg: AUVMjlabCfg,
    entity_name: str,
) -> UnderwaterHydroActionCfg:
    """Build an `UnderwaterHydroActionCfg` from a shared AUV config."""

    return UnderwaterHydroActionCfg(
        entity_name=entity_name,
        body_name=auv_cfg.body_name,
        linear_damping_matrix_6x6=auv_cfg.linear_damping_matrix_6x6,
        quadratic_damping_matrix_6x6=auv_cfg.quadratic_damping_matrix_6x6,
        current_velocity_w=auv_cfg.current_velocity_w,
        current_velocity_b=auv_cfg.current_velocity_b,
        fluid_density_kg_m3=auv_cfg.fluid_density_kg_m3,
        gravity_m_s2=auv_cfg.gravity_m_s2,
        displaced_volume_m3=auv_cfg.displaced_volume_m3,
        buoyancy_n=auv_cfg.buoyancy_n,
        center_of_buoyancy_b_m=auv_cfg.center_of_buoyancy_b_m,
        added_mass_6x6=auv_cfg.added_mass_6x6,
        include_damping=auv_cfg.include_damping,
        include_restoring=auv_cfg.include_restoring,
        include_added_mass=auv_cfg.include_added_mass,
        include_added_coriolis=auv_cfg.include_added_coriolis,
    )


class UnderwaterHydroAction(ActionTerm):
    """Apply the full hydrodynamic wrench at every physics substep.

    Rigid-body mass and center of gravity come directly from the MuJoCo body
    inertial data, so the restoring term and COM wrench shift always match the
    simulated body. Added-mass acceleration is estimated from the finite
    difference of the relative body twist because the action path does not
    expose body accelerations directly.
    """

    cfg: UnderwaterHydroActionCfg
    _entity: Entity

    def __init__(self, cfg: UnderwaterHydroActionCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg=cfg, env=env)

        body_ids, _ = self._entity.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body match for '{cfg.body_name}', got "
                f"{body_ids}."
            )
        self._body_id = int(body_ids[0])
        self._global_body_id = int(
            self._entity.data.indexing.body_ids[self._body_id].item()
        )

        self._raw_actions = torch.empty((self.num_envs, 0), device=self.device)
        self._linear_damping_matrix = torch.as_tensor(
            cfg.linear_damping_matrix_6x6,
            device=self.device,
            dtype=torch.float32,
        )
        self._quadratic_damping_matrix = torch.as_tensor(
            cfg.quadratic_damping_matrix_6x6,
            device=self.device,
            dtype=torch.float32,
        )
        self._added_mass = torch.as_tensor(
            cfg.added_mass_6x6,
            device=self.device,
            dtype=torch.float32,
        )
        self._buoyancy_n = (
            float(cfg.buoyancy_n)
            if cfg.buoyancy_n is not None
            else float(
                cfg.fluid_density_kg_m3 * cfg.gravity_m_s2 * cfg.displaced_volume_m3
            )
        )
        self._current_velocity_w = (
            torch.as_tensor(
                cfg.current_velocity_w,
                device=self.device,
                dtype=torch.float32,
            )
            .view(1, 3)
            .expand(self.num_envs, -1)
            .clone()
        )
        current_velocity_b = (
            (0.0, 0.0, 0.0)
            if cfg.current_velocity_b is None
            else cfg.current_velocity_b
        )
        self._current_velocity_b = (
            torch.as_tensor(
                current_velocity_b,
                device=self.device,
                dtype=torch.float32,
            )
            .view(1, 3)
            .expand(self.num_envs, -1)
            .clone()
        )
        self._current_velocity_is_body = torch.full(
            (self.num_envs,),
            cfg.current_velocity_b is not None,
            dtype=torch.bool,
            device=self.device,
        )
        self._cob_b = torch.as_tensor(
            cfg.center_of_buoyancy_b_m,
            device=self.device,
            dtype=torch.float32,
        ).view(1, 3)
        self._applied_wrench_b = torch.zeros((self.num_envs, 6), device=self.device)
        self._relative_twist_b = torch.zeros((self.num_envs, 6), device=self.device)
        self._has_relative_twist = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def current_velocity_w(self) -> torch.Tensor:
        """Return world-frame current velocity for each environment."""
        quat_wb = self._entity.data.body_link_quat_w[:, self._body_id]
        return self._current_velocity_world(quat_wb)

    @property
    def current_velocity_b(self) -> torch.Tensor:
        """Return body-frame current velocity for each environment."""
        quat_wb = self._entity.data.body_link_quat_w[:, self._body_id]
        return self._current_velocity_body(quat_wb)

    @property
    def applied_wrench_b(self) -> torch.Tensor:
        """Return the last body-frame wrench about the body-frame origin."""
        return self._applied_wrench_b

    def process_actions(self, actions: torch.Tensor) -> None:
        del actions

    def apply_actions(self) -> None:
        quat_wb = self._entity.data.body_link_quat_w[:, self._body_id]
        lin_vel_w = self._entity.data.body_link_lin_vel_w[:, self._body_id]
        ang_vel_w = self._entity.data.body_link_ang_vel_w[:, self._body_id]

        lin_vel_b = quat_apply_inverse(quat_wb, lin_vel_w)
        ang_vel_b = quat_apply_inverse(quat_wb, ang_vel_w)
        current_velocity_b = self._current_velocity_body(quat_wb)

        relative_twist_b = torch.cat((lin_vel_b - current_velocity_b, ang_vel_b), dim=1)
        relative_twist_dot_b = self._relative_twist_dot(relative_twist_b)

        center_of_gravity_b = self._entity.data.model.body_ipos[:, self._global_body_id]
        weight_n = self._entity.data.model.body_mass[:, self._global_body_id] * float(
            self.cfg.gravity_m_s2
        )

        results = compute_hydrodynamic_wrench(
            quat_wxyz=quat_wb,
            relative_twist_b=relative_twist_b,
            relative_twist_dot_b=(
                relative_twist_dot_b if self.cfg.include_added_mass else None
            ),
            linear_damping_matrix_6x6=self._linear_damping_matrix,
            quadratic_damping_matrix_6x6=self._quadratic_damping_matrix,
            added_mass_6x6=self._added_mass,
            center_of_buoyancy_b_m=self._cob_b,
            center_of_gravity_b_m=center_of_gravity_b,
            buoyancy_n=self._buoyancy_n,
            weight_n=weight_n,
            include_damping=self.cfg.include_damping,
            include_restoring=self.cfg.include_restoring,
            include_added_mass=self.cfg.include_added_mass,
            include_added_coriolis=self.cfg.include_added_coriolis,
        )
        wrench_origin_b = results["tau_total_body"]
        wrench_com_b = shift_wrench_origin_to_com(
            wrench_origin_b,
            center_of_gravity_b,
        )

        force_w = quat_apply(quat_wb, wrench_com_b[:, 0:3])
        torque_w = quat_apply(quat_wb, wrench_com_b[:, 3:6])

        self._applied_wrench_b[:] = wrench_origin_b
        self._relative_twist_b[:] = relative_twist_b
        self._has_relative_twist[:] = True

        self._entity.write_external_wrench_to_sim(
            force_w.unsqueeze(1),
            torque_w.unsqueeze(1),
            body_ids=[self._body_id],
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._applied_wrench_b[env_ids] = 0.0
        self._invalidate_relative_twist(env_ids)

        num_envs = self._num_selected_envs(env_ids)
        zeros = torch.zeros((num_envs, 1, 3), device=self.device)
        self._entity.write_external_wrench_to_sim(
            zeros,
            zeros,
            env_ids=env_ids,
            body_ids=[self._body_id],
        )

    def set_current_velocity_w(
        self,
        velocity_w: Sequence[float] | torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set world-frame current velocity for all or selected environments."""
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            num_envs = self._current_velocity_w[env_ids].shape[0]
            self._current_velocity_w[env_ids] = self._expand_velocity(
                velocity_w,
                num_envs,
            )
            self._current_velocity_is_body[env_ids] = False
            self._invalidate_relative_twist(env_ids)
            return

        num_envs = int(env_ids.numel())
        self._current_velocity_w[env_ids] = self._expand_velocity(velocity_w, num_envs)
        self._current_velocity_is_body[env_ids] = False
        self._invalidate_relative_twist(env_ids)

    def set_current_velocity_b(
        self,
        velocity_b: Sequence[float] | torch.Tensor,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        """Set body-frame current velocity for all or selected environments."""
        if env_ids is None:
            env_ids = slice(None)

        if isinstance(env_ids, slice):
            num_envs = self._current_velocity_b[env_ids].shape[0]
            self._current_velocity_b[env_ids] = self._expand_velocity(
                velocity_b,
                num_envs,
            )
            self._current_velocity_is_body[env_ids] = True
            self._invalidate_relative_twist(env_ids)
            return

        num_envs = int(env_ids.numel())
        self._current_velocity_b[env_ids] = self._expand_velocity(velocity_b, num_envs)
        self._current_velocity_is_body[env_ids] = True
        self._invalidate_relative_twist(env_ids)

    def _num_selected_envs(self, env_ids: torch.Tensor | slice | None) -> int:
        if env_ids is None:
            return self.num_envs
        if isinstance(env_ids, slice):
            return self._raw_actions[env_ids].shape[0]
        return int(env_ids.numel())

    def _expand_velocity(
        self,
        velocity: Sequence[float] | torch.Tensor,
        num_envs: int,
    ) -> torch.Tensor:
        tensor = torch.as_tensor(velocity, device=self.device, dtype=torch.float32)
        if tensor.ndim == 1 and tensor.shape[0] == 3:
            return tensor.view(1, 3).expand(num_envs, -1)
        if tensor.ndim == 2 and tensor.shape == (num_envs, 3):
            return tensor
        raise ValueError(
            "velocity must have shape (3,) or "
            f"({num_envs}, 3), got {tuple(tensor.shape)}"
        )

    def _current_velocity_body(self, quat_wb: torch.Tensor) -> torch.Tensor:
        current_velocity_b = quat_apply_inverse(quat_wb, self._current_velocity_w)
        if not bool(torch.any(self._current_velocity_is_body)):
            return current_velocity_b

        current_velocity_b = current_velocity_b.clone()
        current_velocity_b[self._current_velocity_is_body] = self._current_velocity_b[
            self._current_velocity_is_body
        ]
        return current_velocity_b

    def _current_velocity_world(self, quat_wb: torch.Tensor) -> torch.Tensor:
        if not bool(torch.any(self._current_velocity_is_body)):
            return self._current_velocity_w

        current_velocity_w = self._current_velocity_w.clone()
        current_velocity_w[self._current_velocity_is_body] = quat_apply(
            quat_wb[self._current_velocity_is_body],
            self._current_velocity_b[self._current_velocity_is_body],
        )
        return current_velocity_w

    def _relative_twist_dot(self, relative_twist_b: torch.Tensor) -> torch.Tensor:
        relative_twist_dot_b = torch.zeros_like(relative_twist_b)
        if bool(torch.any(self._has_relative_twist)):
            dt_s = float(self._env.physics_dt)
            relative_twist_dot_b[self._has_relative_twist] = (
                relative_twist_b[self._has_relative_twist]
                - self._relative_twist_b[self._has_relative_twist]
            ) / dt_s
        return relative_twist_dot_b

    def _invalidate_relative_twist(
        self,
        env_ids: torch.Tensor | slice | None = None,
    ) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._relative_twist_b[env_ids] = 0.0
        self._has_relative_twist[env_ids] = False
