"""Live odometry panel helpers for Taluy Viser demos."""

from __future__ import annotations

from collections.abc import Callable
import html
import math
from typing import Any

import torch

from mjlab.envs import ManagerBasedRlEnv


def quat_wxyz_to_euler_deg(quat_wxyz: torch.Tensor) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz.detach().cpu().tolist()]

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


def _html_table(title: str, rows: dict[str, str]) -> str:
    row_html = "".join(
        "<tr>"
        f"<td style='padding:0.08rem 0.85rem 0.08rem 0;color:#94a3b8;'>{html.escape(key)}</td>"
        f"<td style='padding:0.08rem 0;color:#e2e8f0;'>{html.escape(value)}</td>"
        "</tr>"
        for key, value in rows.items()
    )
    return (
        f"<div style='margin-top:0.45rem;color:#f8fafc;font-weight:700;'>{html.escape(title)}</div>"
        "<table style='border-collapse:collapse;font-size:0.82rem;'>"
        f"{row_html}"
        "</table>"
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

        pos_w = robot.data.root_link_pos_w[0].detach().cpu().tolist()
        quat_wxyz = robot.data.root_link_quat_w[0]
        rpy_deg = quat_wxyz_to_euler_deg(quat_wxyz)
        lin_vel_w = robot.data.root_link_lin_vel_w[0].detach().cpu().tolist()
        ang_vel_w_deg = [
            math.degrees(float(value))
            for value in robot.data.root_link_ang_vel_w[0].detach().cpu().tolist()
        ]
        lin_vel_b = robot.data.root_link_lin_vel_b[0].detach().cpu().tolist()
        ang_vel_b_deg = [
            math.degrees(float(value))
            for value in robot.data.root_link_ang_vel_b[0].detach().cpu().tolist()
        ]

        command = self._base_env.command_manager.get_command("body_velocity")
        command_values: list[float] | None = None
        if command is not None:
            command_values = command[0].detach().cpu().tolist()

        pose_rows = {
            "position_w_m": format_vec(pos_w),
            "rpy_deg": format_vec(rpy_deg, 2),
        }
        twist_rows = {
            "linear_w_m_s": format_vec(lin_vel_w),
            "angular_w_deg_s": format_vec(ang_vel_w_deg, 2),
            "linear_b_m_s": format_vec(lin_vel_b),
            "angular_b_deg_s": format_vec(ang_vel_b_deg, 2),
        }
        if command_values is not None:
            twist_rows["command_b"] = (
                f"{format_vec(command_values[:3])} m/s, "
                f"{format_vec([math.degrees(v) for v in command_values[3:]], 2)} deg/s"
            )

        self._html.content = (
            "<div style='padding:0.55rem;font-family:monospace;"
            "background:rgba(15,23,42,0.94);border:1px solid #334155;"
            "border-radius:6px;color:#f8fafc;line-height:1.35;'>"
            "<div style='font-weight:800;font-size:1rem;'>Odometry</div>"
            + _html_table("Pose", pose_rows)
            + _html_table("Twist", twist_rows)
            + "</div>"
        )
