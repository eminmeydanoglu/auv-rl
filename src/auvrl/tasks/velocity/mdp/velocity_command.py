from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from auvrl.actuator.body_wrench_action import BodyWrenchAction
from auvrl.actuator.thruster_actuator import THRUSTER_LOCAL_AXIS
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply

if TYPE_CHECKING:
    import viser

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer


def _validate_range(name: str, value: tuple[float, float]) -> None:
    if float(value[1]) < float(value[0]):
        raise ValueError(f"{name} must satisfy lower <= upper, got {value}.")


def _offset_in_world(
    base_pos_w: np.ndarray,
    base_mat_w: np.ndarray,
    offset_b: tuple[float, float, float],
) -> np.ndarray:
    return base_pos_w + base_mat_w @ np.asarray(offset_b, dtype=float)


def _add_body_vector_indicator(
    visualizer: "DebugVisualizer",
    *,
    origin_w: np.ndarray,
    base_mat_w: np.ndarray,
    vector_b: np.ndarray,
    scale: float,
    color: tuple[float, float, float, float],
    width: float,
    zero_threshold: float,
    zero_marker_radius: float,
) -> None:
    if np.linalg.norm(vector_b) < zero_threshold:
        visualizer.add_sphere(
            center=origin_w,
            radius=zero_marker_radius,
            color=(color[0], color[1], color[2], min(color[3], 0.55)),
        )
        return

    end_w = origin_w + base_mat_w @ (vector_b * scale)
    visualizer.add_arrow(origin_w, end_w, color=color, width=width)


class UniformBodyVelocityCommand(CommandTerm):
    cfg: UniformBodyVelocityCommandCfg

    def __init__(self, cfg: UniformBodyVelocityCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)

        self.robot: Entity = env.scene[cfg.entity_name]
        self.vel_command_b = torch.zeros(self.num_envs, 6, device=self.device)
        self.is_zero_env = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )

        self.metrics["error_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_ang_vel"] = torch.zeros(self.num_envs, device=self.device)

        self._joystick_enabled: viser.GuiCheckboxHandle | None = None
        self._joystick_sliders: list[viser.GuiSliderHandle] = []
        self._joystick_get_env_idx: Callable[[], int] | None = None

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    def _update_metrics(self) -> None:
        max_command_time = max(
            float(self.cfg.resampling_time_range[1]), self._env.step_dt
        )
        max_command_steps = max(max_command_time / max(self._env.step_dt, 1.0e-6), 1.0)

        lin_error = torch.linalg.norm(
            self.vel_command_b[:, :3] - self.robot.data.root_link_lin_vel_b,
            dim=1,
        )
        ang_error = torch.linalg.norm(
            self.vel_command_b[:, 3:] - self.robot.data.root_link_ang_vel_b,
            dim=1,
        )
        self.metrics["error_lin_vel"] += lin_error / max_command_steps
        self.metrics["error_ang_vel"] += ang_error / max_command_steps

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return

        sample = torch.empty(len(env_ids), device=self.device)
        ranges = self.cfg.ranges

        self.vel_command_b[env_ids, 0] = sample.uniform_(*ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = sample.uniform_(*ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = sample.uniform_(*ranges.lin_vel_z)
        self.vel_command_b[env_ids, 3] = sample.uniform_(*ranges.ang_vel_x)
        self.vel_command_b[env_ids, 4] = sample.uniform_(*ranges.ang_vel_y)
        self.vel_command_b[env_ids, 5] = sample.uniform_(*ranges.ang_vel_z)

        if self.cfg.rel_zero_envs > 0.0:
            zero_mask = (
                torch.rand(len(env_ids), device=self.device) < self.cfg.rel_zero_envs
            )
            zero_env_ids = env_ids[zero_mask]
            if len(zero_env_ids) > 0:
                self.vel_command_b[zero_env_ids, :] = 0.0
            self.is_zero_env[env_ids] = zero_mask
        else:
            self.is_zero_env[env_ids] = False

        if self.cfg.init_velocity_prob > 0.0:
            init_mask = (
                torch.rand(len(env_ids), device=self.device)
                < self.cfg.init_velocity_prob
            )
            init_env_ids = env_ids[init_mask]
            if len(init_env_ids) > 0:
                root_quat = self.robot.data.root_link_quat_w[init_env_ids]
                lin_vel_w = quat_apply(root_quat, self.vel_command_b[init_env_ids, :3])
                ang_vel_w = quat_apply(root_quat, self.vel_command_b[init_env_ids, 3:])
                root_vel_w = torch.cat((lin_vel_w, ang_vel_w), dim=-1)
                self.robot.write_root_link_velocity_to_sim(root_vel_w, init_env_ids)

    def _update_command(self) -> None:
        pass

    def create_gui(
        self,
        name: str,
        server: "viser.ViserServer",
        get_env_idx: Callable[[], int],
    ) -> None:
        from viser import Icon

        axes = [
            ("lin_vel_x", self.cfg.ranges.lin_vel_x),
            ("lin_vel_y", self.cfg.ranges.lin_vel_y),
            ("lin_vel_z", self.cfg.ranges.lin_vel_z),
            ("ang_vel_x", self.cfg.ranges.ang_vel_x),
            ("ang_vel_y", self.cfg.ranges.ang_vel_y),
            ("ang_vel_z", self.cfg.ranges.ang_vel_z),
        ]
        sliders: list[viser.GuiSliderHandle] = []

        with server.gui.add_folder(name.replace("_", " ").title()):
            enabled = server.gui.add_checkbox("Enable", initial_value=False)
            for label, axis_range in axes:
                max_mag = max(abs(float(axis_range[0])), abs(float(axis_range[1])), 0.1)
                max_input = server.gui.add_slider(
                    f"Max {label}",
                    initial_value=max_mag,
                    step=0.1,
                    min=0.1,
                    max=10.0,
                )
                slider = server.gui.add_slider(
                    label,
                    min=-max_mag,
                    max=max_mag,
                    step=0.05,
                    initial_value=0.0,
                )

                @max_input.on_update
                def _(_ev, _slider=slider, _max_input=max_input) -> None:
                    _slider.min = -_max_input.value
                    _slider.max = _max_input.value

                sliders.append(slider)

            zero_btn = server.gui.add_button("Zero", icon=Icon.SQUARE_X)

            @zero_btn.on_click
            def _(_) -> None:
                for slider in sliders:
                    slider.value = 0.0

        self._joystick_enabled = enabled
        self._joystick_sliders = sliders
        self._joystick_get_env_idx = get_env_idx

    def compute(self, dt: float) -> None:
        super().compute(dt)
        if self._joystick_enabled is not None and self._joystick_enabled.value:
            assert self._joystick_get_env_idx is not None
            env_idx = self._joystick_get_env_idx()
            for i, slider in enumerate(self._joystick_sliders):
                self.vel_command_b[env_idx, i] = slider.value

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return

        commands = self.command.cpu().numpy()
        base_pos_ws = self.robot.data.root_link_pos_w.cpu().numpy()
        base_quat_w = self.robot.data.root_link_quat_w
        base_mat_ws = matrix_from_quat(base_quat_w).cpu().numpy()
        lin_vel_bs = self.robot.data.root_link_lin_vel_b.cpu().numpy()

        viz = self.cfg.viz
        linear_scale = viz.linear_scale
        try:
            wrench_term = self._env.action_manager.get_term("body_wrench")
        except KeyError:
            wrench_term = None
        thruster_site_pos_ws: np.ndarray | None = None
        thruster_force_axis_ws: np.ndarray | None = None
        thruster_targets: np.ndarray | None = None
        if isinstance(wrench_term, BodyWrenchAction):
            site_ids = wrench_term.site_ids
            num_sites = int(site_ids.numel())
            thruster_site_pos_ws = (
                self.robot.data.data.site_xpos[:, site_ids].detach().cpu().numpy()
            )
            thruster_site_mat_ws = (
                self.robot.data.data.site_xmat[:, site_ids]
                .reshape(self.num_envs, num_sites, 3, 3)
                .detach()
                .cpu()
                .numpy()
            )
            thruster_force_axis_ws = thruster_site_mat_ws @ np.asarray(
                THRUSTER_LOCAL_AXIS,
                dtype=float,
            )
            thruster_targets = wrench_term.thruster_targets.detach().cpu().numpy()

        for env_idx in env_indices:
            base_pos_w = base_pos_ws[env_idx]
            base_mat_w = base_mat_ws[env_idx]
            command = commands[env_idx]
            lin_vel_b = lin_vel_bs[env_idx]

            if np.linalg.norm(base_pos_w) < 1e-6:
                continue

            _add_body_vector_indicator(
                visualizer,
                origin_w=_offset_in_world(
                    base_pos_w, base_mat_w, viz.command_linear_offset_b
                ),
                base_mat_w=base_mat_w,
                vector_b=np.asarray(command[:3], dtype=float),
                scale=linear_scale,
                color=(0.75, 0.3, 0.95, 0.9),
                width=viz.command_vector_width,
                zero_threshold=viz.linear_zero_threshold,
                zero_marker_radius=viz.zero_marker_radius,
            )

            _add_body_vector_indicator(
                visualizer,
                origin_w=_offset_in_world(
                    base_pos_w, base_mat_w, viz.measured_linear_offset_b
                ),
                base_mat_w=base_mat_w,
                vector_b=np.asarray(lin_vel_b, dtype=float),
                scale=linear_scale,
                color=(0.05, 0.7, 1.0, 0.9),
                width=viz.measured_vector_width,
                zero_threshold=viz.linear_zero_threshold,
                zero_marker_radius=viz.zero_marker_radius,
            )

            if (
                thruster_site_pos_ws is None
                or thruster_force_axis_ws is None
                or thruster_targets is None
            ):
                continue

            for thruster_idx, force_n in enumerate(thruster_targets[env_idx]):
                force_n = float(force_n)
                if abs(force_n) < viz.thruster_force_zero_threshold_n:
                    continue

                start = thruster_site_pos_ws[env_idx, thruster_idx]
                end = start + (
                    thruster_force_axis_ws[env_idx, thruster_idx]
                    * (force_n * viz.thruster_force_scale)
                )
                color = (
                    (1.0, 0.65, 0.1, 0.85) if force_n >= 0.0 else (1.0, 0.35, 0.6, 0.85)
                )
                visualizer.add_arrow(
                    start,
                    end,
                    color=color,
                    width=viz.thruster_force_width,
                )


@dataclass(kw_only=True)
class UniformBodyVelocityCommandCfg(CommandTermCfg):
    entity_name: str
    rel_zero_envs: float = 0.0
    init_velocity_prob: float = 0.0

    @dataclass
    class Ranges:
        lin_vel_x: tuple[float, float]
        lin_vel_y: tuple[float, float]
        lin_vel_z: tuple[float, float]
        ang_vel_x: tuple[float, float]
        ang_vel_y: tuple[float, float]
        ang_vel_z: tuple[float, float]

    ranges: Ranges

    @dataclass
    class VizCfg:
        linear_scale: float = 0.8
        command_vector_width: float = 0.02
        measured_vector_width: float = 0.012
        zero_marker_radius: float = 0.02
        linear_zero_threshold: float = 0.02
        thruster_force_scale: float = 0.015
        thruster_force_width: float = 0.012
        thruster_force_zero_threshold_n: float = 1.0
        command_linear_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.30)
        measured_linear_offset_b: tuple[float, float, float] = (0.0, 0.0, 0.30)

    viz: VizCfg = field(default_factory=VizCfg)

    def build(self, env: ManagerBasedRlEnv) -> UniformBodyVelocityCommand:
        return UniformBodyVelocityCommand(self, env)

    def __post_init__(self) -> None:
        _validate_range("resampling_time_range", self.resampling_time_range)
        _validate_range("ranges.lin_vel_x", self.ranges.lin_vel_x)
        _validate_range("ranges.lin_vel_y", self.ranges.lin_vel_y)
        _validate_range("ranges.lin_vel_z", self.ranges.lin_vel_z)
        _validate_range("ranges.ang_vel_x", self.ranges.ang_vel_x)
        _validate_range("ranges.ang_vel_y", self.ranges.ang_vel_y)
        _validate_range("ranges.ang_vel_z", self.ranges.ang_vel_z)
        if not 0.0 <= float(self.rel_zero_envs) <= 1.0:
            raise ValueError(
                f"rel_zero_envs must be in [0, 1], got {self.rel_zero_envs}."
            )
        if not 0.0 <= float(self.init_velocity_prob) <= 1.0:
            raise ValueError(
                f"init_velocity_prob must be in [0, 1], got {self.init_velocity_prob}."
            )
