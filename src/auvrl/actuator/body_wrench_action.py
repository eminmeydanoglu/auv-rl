"""Body-frame wrench action term with thruster allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.managers.action_manager import ActionTerm, ActionTermCfg

from .thruster_allocation import allocation_matrix_from_mujoco_sites
from .thruster_actuator import THRUSTER_LOCAL_AXIS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BodyWrenchActionCfg(ActionTermCfg):
    """Map body-frame wrench commands to site thruster force targets.

    The policy action vector is ``[Fx, Fy, Fz, Mx, My, Mz]`` in normalized
    body-frame units. Each component is clipped to ``[-1, 1]`` and then scaled
    by ``wrench_limit`` to recover the wrench in ``[N, N, N, N*m, N*m, N*m]``
    before allocation.

    Thruster geometry is inferred from MuJoCo site poses assuming the vehicle
    convention that positive thrust acts along each site's local ``-Z`` axis.

    When ``neutralize_com_coupling`` is enabled, torque commands are interpreted
    about the body's center of mass and are shifted to the MuJoCo body origin
    before allocation. This keeps pure force commands from inheriting unwanted
    moment from a non-zero COM offset.
    """

    body_name: str
    actuator_names: tuple[str, ...]
    wrench_limit: tuple[float, float, float, float, float, float]
    preserve_order: bool = False
    neutralize_com_coupling: bool = True
    require_full_rank: bool = True
    site_force_limit_n: float | None = None

    def __post_init__(self) -> None:
        if len(self.actuator_names) == 0:
            raise ValueError("actuator_names must contain at least one site pattern.")
        if len(self.wrench_limit) != 6:
            raise ValueError("wrench_limit must contain exactly 6 values.")
        if any(float(value) <= 0.0 for value in self.wrench_limit):
            raise ValueError("wrench_limit values must all be positive.")
        if (
            self.site_force_limit_n is not None
            and float(self.site_force_limit_n) <= 0.0
        ):
            raise ValueError("site_force_limit_n must be positive when provided.")

    def build(self, env: ManagerBasedRlEnv) -> BodyWrenchAction:
        return BodyWrenchAction(self, env)


class BodyWrenchAction(ActionTerm):
    """Allocate body-frame wrench commands to thruster site targets."""

    cfg: BodyWrenchActionCfg

    def __init__(self, cfg: BodyWrenchActionCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg=cfg, env=env)

        body_ids, _ = self._entity.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected exactly one body match for '{cfg.body_name}', got {body_ids}."
            )
        self._body_id = int(body_ids[0])
        self._global_body_id = int(
            self._entity.data.indexing.body_ids[self._body_id].item()
        )

        site_ids, site_names = self._entity.find_sites(
            cfg.actuator_names,
            preserve_order=cfg.preserve_order,
        )
        if len(site_ids) == 0:
            raise ValueError(
                "BodyWrenchAction could not resolve any thruster sites from "
                f"{cfg.actuator_names}."
            )

        allocation_matrix = allocation_matrix_from_mujoco_sites(
            model=env.sim.mj_model,
            data=env.sim.mj_data,
            body_name=cfg.body_name,
            site_names=site_names,
            local_force_axis=THRUSTER_LOCAL_AXIS,
        )
        allocation = torch.as_tensor(
            allocation_matrix,
            dtype=torch.float,
            device=self.device,
        )
        if allocation.shape[0] != 6:
            raise ValueError(
                "Allocation matrix must have 6 wrench rows, got "
                f"shape {tuple(allocation.shape)}."
            )
        rank = int(torch.linalg.matrix_rank(allocation).item())
        if cfg.require_full_rank and rank < 6:
            raise ValueError(
                "Allocation matrix is not full-rank for 6D wrench control. "
                f"Observed rank {rank} for shape {tuple(allocation.shape)}."
            )

        self._site_ids = torch.tensor(site_ids, dtype=torch.long, device=self.device)
        self._site_names = tuple(site_names)
        self._allocation_matrix_b = allocation
        self._allocation_pinv_t = torch.linalg.pinv(allocation).transpose(0, 1)

        self._wrench_limit = torch.tensor(
            cfg.wrench_limit,
            dtype=torch.float,
            device=self.device,
        ).view(1, 6)

        self._site_force_limit_n = (
            None if cfg.site_force_limit_n is None else float(cfg.site_force_limit_n)
        )

        self._raw_actions = torch.zeros((self.num_envs, 6), device=self.device)
        self._desired_wrench_b = torch.zeros_like(self._raw_actions)
        self._applied_wrench_origin_b = torch.zeros_like(self._raw_actions)
        self._thruster_targets = torch.zeros(
            (self.num_envs, len(site_ids)),
            device=self.device,
        )
        self._step_saturation_fraction = torch.zeros(self.num_envs, device=self.device)

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def desired_wrench_b(self) -> torch.Tensor:
        """Return clipped body-frame wrench command before COM shift."""
        return self._desired_wrench_b

    @property
    def applied_wrench_origin_b(self) -> torch.Tensor:
        """Return body-frame wrench actually allocated about the body origin."""
        return self._applied_wrench_origin_b

    @property
    def thruster_targets(self) -> torch.Tensor:
        """Return commanded per-thruster force targets in newtons."""
        return self._thruster_targets

    @property
    def site_ids(self) -> torch.Tensor:
        """Return resolved MuJoCo site ids for the controlled thrusters."""
        return self._site_ids

    @property
    def step_saturation_fraction(self) -> torch.Tensor:
        """Return fraction of thrusters clipped by the site-force limit."""
        return self._step_saturation_fraction

    @property
    def allocation_matrix_b(self) -> torch.Tensor:
        """Return the body-origin 6xN thrust allocation matrix."""
        return self._allocation_matrix_b

    @property
    def wrench_limit(self) -> torch.Tensor:
        """Return body-wrench limits used to scale normalized policy actions."""
        return self._wrench_limit

    def process_actions(self, actions: torch.Tensor) -> None:
        effective_action = self._clip_policy_action(actions)
        self._raw_actions[:] = effective_action
        self._desired_wrench_b[:] = self.action_to_wrench(effective_action)

    def apply_actions(self) -> None:
        applied_wrench_b = self._desired_wrench_b
        if self.cfg.neutralize_com_coupling:
            applied_wrench_b = self._shift_com_wrench_to_body_origin(
                self._desired_wrench_b
            )

        thruster_targets = applied_wrench_b @ self._allocation_pinv_t
        if self._site_force_limit_n is None:
            self._step_saturation_fraction.zero_()
        else:
            limit = self._site_force_limit_n
            thruster_targets = thruster_targets.clamp(min=-limit, max=limit)
            saturated = thruster_targets.abs() >= (limit - 1.0e-6)
            self._step_saturation_fraction[:] = saturated.float().mean(dim=1)

        self._applied_wrench_origin_b[:] = applied_wrench_b
        self._thruster_targets[:] = thruster_targets
        self._entity.set_site_effort_target(
            self._thruster_targets,
            site_ids=self._site_ids,
        )

    def action_to_wrench(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert policy-space actions to clipped body-frame wrench commands."""
        effective_action = self._clip_policy_action(actions)
        return effective_action * self._wrench_limit

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._raw_actions[env_ids] = 0.0
        self._desired_wrench_b[env_ids] = 0.0
        self._applied_wrench_origin_b[env_ids] = 0.0
        self._thruster_targets[env_ids] = 0.0
        self._step_saturation_fraction[env_ids] = 0.0

        if isinstance(env_ids, torch.Tensor):
            self._entity.data.site_effort_target[env_ids[:, None], self._site_ids] = 0.0
            return

        num_envs = self._num_selected_envs(env_ids)
        zeros = torch.zeros((num_envs, len(self._site_names)), device=self.device)
        self._entity.set_site_effort_target(
            zeros, site_ids=self._site_ids, env_ids=env_ids
        )

    def _shift_com_wrench_to_body_origin(self, wrench_b: torch.Tensor) -> torch.Tensor:
        force_b = wrench_b[:, :3]
        torque_com_b = wrench_b[:, 3:]
        com_offset_b = self._entity.data.model.body_ipos[:, self._global_body_id]
        torque_origin_b = torque_com_b + torch.cross(com_offset_b, force_b, dim=1)
        return torch.cat((force_b, torque_origin_b), dim=1)

    def _clip_policy_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.clamp(min=-1.0, max=1.0)

    def _num_selected_envs(self, env_ids: torch.Tensor | slice | None) -> int:
        if env_ids is None:
            return self.num_envs
        if isinstance(env_ids, slice):
            return self._raw_actions[env_ids].shape[0]
        return int(env_ids.numel())
