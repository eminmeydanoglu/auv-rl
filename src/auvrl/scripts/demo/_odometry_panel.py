"""Live odometry panel helpers for Taluy Viser demos."""

from __future__ import annotations

from collections.abc import Callable
import html
import math
from typing import Any

import torch

from auvrl.actuator.body_wrench_action import BodyWrenchAction
from auvrl.sim.underwater_hydro_action import UnderwaterHydroAction
from auvrl.tasks.velocity.mdp import UniformBodyVelocityCommand
from mjlab.envs import ManagerBasedRlEnv


def _tensor_to_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


def quat_wxyz_to_euler_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    w, x, y, z = _tensor_to_list(quat_wxyz)

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def format_vec(values: list[float] | tuple[float, ...], precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _wrench_label(wrench: list[float], precision: int = 2) -> str:
    return (
        f"F[N]{format_vec(wrench[:3], precision)} "
        f"M[Nm]{format_vec(wrench[3:], precision)}"
    )


def _command_gui_enabled(base_env: ManagerBasedRlEnv) -> bool:
    try:
        term = base_env.command_manager.get_term("body_velocity")
    except KeyError:
        return True
    if isinstance(term, UniformBodyVelocityCommand):
        return term.gui_enabled
    return True


def _html_section(
    title: str,
    rows: dict[str, str],
    *,
    open_by_default: bool = True,
) -> str:
    if not rows:
        return ""
    row_html = "".join(
        "<div style='display:grid;grid-template-columns:7.7rem max-content;"
        "column-gap:0.65rem;align-items:baseline;white-space:nowrap;"
        "padding:0.06rem 0;'>"
        f"<span style='color:#94a3b8;'>{html.escape(key)}</span>"
        f"<span style='color:#e2e8f0;'>{html.escape(value)}</span>"
        "</div>"
        for key, value in rows.items()
    )
    open_attr = " open" if open_by_default else ""
    return (
        f"<details{open_attr} style='margin-top:0.45rem;'>"
        "<summary style='cursor:pointer;color:#f8fafc;font-weight:800;"
        "list-style-position:outside;'>"
        f"{html.escape(title)}</summary>"
        "<div style='margin-top:0.18rem;'>"
        f"{row_html}"
        "</div>"
        "</details>"
    )


class CommandEnableGatedPolicy:
    """Return zero actions unless the Body Velocity GUI Enable box is checked."""

    def __init__(
        self,
        base_policy: Callable[[Any], torch.Tensor],
        base_env: ManagerBasedRlEnv,
    ) -> None:
        self._base_policy = base_policy
        self._base_env = base_env

    def __call__(self, obs: Any) -> torch.Tensor:
        if _command_gui_enabled(self._base_env):
            return self._base_policy(obs)
        action_dim = int(self._base_env.action_manager.total_action_dim)
        return torch.zeros(
            (self._base_env.num_envs, action_dim),
            device=self._base_env.device,
            dtype=torch.float,
        )


class OdometryTelemetryPolicy:
    """Wrap a policy and publish live pose/twist values to a Viser HTML panel."""

    def __init__(
        self,
        base_policy: Callable[[Any], torch.Tensor],
        base_env: ManagerBasedRlEnv,
        period_steps: int,
        server: Any,
    ) -> None:
        self._base_policy = base_policy
        self._base_env = base_env
        self._period_steps = max(int(period_steps), 1)
        self._step = 0
        self._html = server.gui.add_html("")

    def __call__(self, obs: Any) -> torch.Tensor:
        actions = self._base_policy(obs)
        if self._step % self._period_steps == 0:
            self._update_panel()
        self._step += 1
        return actions

    def _update_panel(self) -> None:
        robot = self._base_env.scene["robot"]

        pos_w = _tensor_to_list(robot.data.root_link_pos_w[0])
        quat_wxyz = robot.data.root_link_quat_w[0]
        rpy_deg = quat_wxyz_to_euler_deg(quat_wxyz)
        lin_vel_w = _tensor_to_list(robot.data.root_link_lin_vel_w[0])
        ang_vel_w_deg = [
            math.degrees(float(value))
            for value in _tensor_to_list(robot.data.root_link_ang_vel_w[0])
        ]
        lin_vel_b = _tensor_to_list(robot.data.root_link_lin_vel_b[0])
        ang_vel_b_deg = [
            math.degrees(float(value))
            for value in _tensor_to_list(robot.data.root_link_ang_vel_b[0])
        ]

        command = self._base_env.command_manager.get_command("body_velocity")
        command_values: list[float] | None = None
        if command is not None:
            command_values = _tensor_to_list(command[0])
        controller_enabled = _command_gui_enabled(self._base_env)

        motor_wrench_b: list[float] | None = None
        desired_motor_wrench_b: list[float] | None = None
        thruster_targets_n: list[float] | None = None
        saturation_fraction: float | None = None
        try:
            body_wrench_term = self._base_env.action_manager.get_term("body_wrench")
        except KeyError:
            body_wrench_term = None
        if isinstance(body_wrench_term, BodyWrenchAction):
            thruster_targets = body_wrench_term.thruster_targets[0]
            motor_wrench = body_wrench_term.allocation_matrix_b @ thruster_targets
            motor_wrench_b = _tensor_to_list(motor_wrench)
            desired_motor_wrench_b = _tensor_to_list(
                body_wrench_term.desired_wrench_b[0]
            )
            thruster_targets_n = _tensor_to_list(thruster_targets)
            saturation_fraction = float(
                body_wrench_term.step_saturation_fraction[0].detach().cpu().item()
            )

        hydro_wrench_b: list[float] | None = None
        try:
            hydro_term = self._base_env.action_manager.get_term("hydro")
        except KeyError:
            hydro_term = None
        if isinstance(hydro_term, UnderwaterHydroAction):
            hydro_wrench_b = _tensor_to_list(hydro_term.applied_wrench_b[0])

        net_wrench_b: list[float] | None = None
        if motor_wrench_b is not None and hydro_wrench_b is not None:
            net_wrench_b = [
                motor_value + hydro_value
                for motor_value, hydro_value in zip(
                    motor_wrench_b,
                    hydro_wrench_b,
                    strict=True,
                )
            ]

        pose_rows = {
            "pos_w_m": format_vec(pos_w),
            "rpy_deg": format_vec(rpy_deg, 2),
        }
        twist_rows = {
            "lin_w_m_s": format_vec(lin_vel_w),
            "ang_w_deg_s": format_vec(ang_vel_w_deg, 2),
            "lin_b_m_s": format_vec(lin_vel_b),
            "ang_b_deg_s": format_vec(ang_vel_b_deg, 2),
        }
        if command_values is not None:
            twist_rows["cmd_b"] = (
                f"{format_vec(command_values[:3])} m/s, "
                f"{format_vec([math.degrees(v) for v in command_values[3:]], 2)} deg/s"
            )

        wrench_rows: dict[str, str] = {}
        wrench_rows["ctrl_enabled"] = "yes" if controller_enabled else "no"
        if desired_motor_wrench_b is not None:
            wrench_rows["motor_des_com_b"] = _wrench_label(desired_motor_wrench_b)
        if motor_wrench_b is not None:
            wrench_rows["motor_act_org_b"] = _wrench_label(motor_wrench_b)
        if hydro_wrench_b is not None:
            wrench_rows["hydro_ext_org_b"] = _wrench_label(hydro_wrench_b)
        if net_wrench_b is not None:
            wrench_rows["net_org_b"] = _wrench_label(net_wrench_b)
        if thruster_targets_n is not None:
            wrench_rows["thr_n"] = format_vec(thruster_targets_n, 1)
        if saturation_fraction is not None:
            wrench_rows["sat"] = f"{saturation_fraction:.3f}"

        self._html.content = (
            "<div style='padding:0.55rem;font-family:monospace;"
            "background:rgba(15,23,42,0.94);border:1px solid #334155;"
            "border-radius:6px;color:#f8fafc;line-height:1.28;"
            "font-size:0.74rem;overflow-x:auto;'>"
            "<div style='font-weight:900;font-size:0.9rem;white-space:nowrap;'>"
            "Telemetry</div>"
            + _html_section("Pose", pose_rows)
            + _html_section("Twist", twist_rows)
            + _html_section("Wrench", wrench_rows)
            + "</div>"
        )
