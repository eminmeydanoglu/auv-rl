"""Interactive Taluy roll-task training inspector.

This launches the deterministic Taluy roll task without loading a neural
network.  The Viser command GUI exposes a manual 6-DoF body-velocity reference,
and a small hand-tuned feedback policy converts that reference into the roll
task's normalized ``body_wrench`` action.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from collections import deque
from datetime import datetime
import html
import json
import math
import os
from pathlib import Path
import time
import traceback
from typing import Any, cast

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
    from mjlab.viewer import (  # type: ignore[import-not-found]
        NativeMujocoViewer,
        VerbosityLevel,
        ViserPlayViewer,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab RL/viewer dependencies. Ensure mjlab is available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    UniformBodyVelocityCommand,
    UniformBodyVelocityCommandCfg,
    make_taluy_roll_env_cfg,
)
from auvrl.actuator.body_wrench_action import (  # noqa: E402  # type: ignore[import-not-found]
    BodyWrenchAction,
)
from auvrl.config.auv_cfg import (  # noqa: E402  # type: ignore[import-not-found]
    TALUY_CFG_PATH,
    load_auv_cfg,
)
from auvrl.tasks.roll.runtime import (  # noqa: E402  # type: ignore[import-not-found]
    current_root_pose_from_qpos,
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_COMMAND = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="viser",
        help="Viewer backend. Use `viser` for the command GUI and inspector panel.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel env count.",
    )
    parser.add_argument(
        "--inspect-env-idx",
        type=int,
        default=0,
        help="Environment index shown in console, JSONL, and the Viser panel.",
    )
    parser.add_argument(
        "--viser-host",
        default="0.0.0.0",
        help="Viser host address.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=9000,
        help="Viser port.",
    )
    parser.add_argument(
        "--print-period-s",
        type=float,
        default=1.0,
        help="Wall-clock update period for console and JSONL telemetry.",
    )
    parser.add_argument(
        "--panel-period-s",
        type=float,
        default=0.2,
        help="Wall-clock update period for the Viser inspector panel.",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=0,
        help="Run a short headless rollout instead of opening a viewer.",
    )
    parser.add_argument(
        "--fixed-command",
        type=float,
        nargs=6,
        default=None,
        metavar=("VX", "VY", "VZ", "WX", "WY", "WZ"),
        help="Fixed body-velocity command for dry-runs or viewer startup.",
    )
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable all terminations for uninterrupted inspection.",
    )
    parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="JSONL telemetry path. Defaults to logs/manual_roll_inspector/<stamp>.jsonl.",
    )
    parser.add_argument(
        "--no-jsonl",
        action="store_true",
        help="Disable persistent JSONL telemetry.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _default_jsonl_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "logs" / "manual_roll_inspector" / f"{stamp}.jsonl"


def _build_viser_server(host: str, port: int):
    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    try:
        return viser.ViserServer(host=host, port=port, label="taluy-roll-inspector")
    except TypeError:
        print("Warning: this viser version does not support host/port arguments.")
        return viser.ViserServer(label="taluy-roll-inspector")


def _make_body_velocity_command_cfg() -> UniformBodyVelocityCommandCfg:
    return UniformBodyVelocityCommandCfg(
        entity_name="robot",
        resampling_time_range=(1.0e6, 1.0e6),
        rel_zero_envs=1.0,
        init_velocity_prob=0.0,
        debug_vis=True,
        ranges=UniformBodyVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.6, 0.6),
            lin_vel_y=(-0.6, 0.6),
            lin_vel_z=(-0.4, 0.4),
            ang_vel_x=(-1.0, 1.0),
            ang_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.2, 1.2),
        ),
    )


def _make_roll_inspector_env_cfg(args: argparse.Namespace):
    cfg = make_taluy_roll_env_cfg(
        num_envs=args.num_envs,
        episode_length_s=300.0,
    )
    cfg.commands["body_velocity"] = _make_body_velocity_command_cfg()
    if args.no_terminations:
        cfg.terminations = {}
    return cfg


def _tensor_env_value(value: torch.Tensor, env_idx: int) -> Any:
    if value.ndim == 0:
        return float(value.detach().cpu().item())
    selected = value[env_idx]
    if selected.ndim == 0:
        item = selected.detach().cpu().item()
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        return float(item)
    return selected.detach().cpu().tolist()


def _iterable_terms_to_dict(
    terms: Sequence[tuple[str, Sequence[float]]],
    *,
    prefix: str | None = None,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, values in terms:
        if prefix is not None and not name.startswith(prefix):
            continue
        key = name if prefix is None else name[len(prefix) :]
        result[key] = [float(value) for value in values]
    return result


def _format_table(title: str, values: dict[str, Any], *, max_items: int | None = None) -> str:
    lines = [f"### {title}", ""]
    selected_items = list(values.items())
    if max_items is not None:
        selected_items = selected_items[:max_items]
    if not selected_items:
        lines.append("_none_")
        return "\n".join(lines)

    for key, value in selected_items:
        lines.append(f"- `{key}`: `{_compact_value(value)}`")
    return "\n".join(lines)


def _compact_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.5g}"
    if isinstance(value, int | bool):
        return str(value)
    if isinstance(value, list | tuple):
        parts = []
        for item in value:
            if isinstance(item, float):
                parts.append(f"{item:.4g}")
            else:
                parts.append(str(item))
        return "[" + ", ".join(parts) + "]"
    if isinstance(value, dict):
        return json.dumps(value, separators=(",", ":"), default=str)
    return str(value)


class ScalarBarPanel:
    def __init__(
        self,
        server: Any,
        title: str,
        *,
        max_rows: int = 48,
        window_s: float = 0.25,
        update_dt_s: float = 0.2,
    ) -> None:
        self._title = title
        self._max_rows = max_rows
        window_steps = max(1, round(window_s / max(update_dt_s, 1.0e-6)))
        self._histories: dict[str, deque[float]] = {}
        self._window_steps = window_steps
        self._html_handle = server.gui.add_html("")
        self._render_empty()

    def update(self, values: dict[str, Any]) -> None:
        flat_values = self._flatten(values)
        for name, value in flat_values.items():
            if name not in self._histories:
                self._histories[name] = deque(maxlen=self._window_steps)
            if math.isfinite(value):
                self._histories[name].append(float(value))
        self._render()

    def clear_histories(self) -> None:
        self._histories.clear()
        self._render_empty()

    def remove(self) -> None:
        self._html_handle.remove()

    def _render_empty(self) -> None:
        self._html_handle.content = (
            f'<div style="padding:0.5em;color:#999;font-size:0.85em;">'
            f"{html.escape(self._title)}: waiting for data...</div>"
        )

    def _render(self) -> None:
        means: dict[str, float] = {}
        for name, history in self._histories.items():
            means[name] = sum(history) / len(history) if history else 0.0

        visible_names = list(means.keys())[: self._max_rows]
        max_abs = max((abs(means[name]) for name in visible_names), default=1.0)
        max_abs = max(max_abs, 1.0e-12)

        rows: list[str] = []
        for name in visible_names:
            value = means[name]
            magnitude_pct = min(abs(value) / max_abs * 50.0, 50.0)
            if value >= 0.0:
                left_width = 0.0
                right_width = magnitude_pct
                color = "#4caf50"
            else:
                left_width = magnitude_pct
                right_width = 0.0
                color = "#f44336"

            value_str = f"{value:.2e}" if 0.0 < abs(value) < 1.0e-4 else f"{value:.4f}"
            safe_name = html.escape(name, quote=True)
            rows.append(
                '<div style="display:flex;align-items:center;margin:2px 0;">'
                f'<span style="min-width:132px;font-size:0.76em;text-align:right;'
                f"padding-right:6px;color:#ddd;white-space:nowrap;overflow:hidden;"
                f'text-overflow:ellipsis;" title="{safe_name}">{safe_name}</span>'
                '<div style="flex:1;background:#303030;border-radius:3px;height:18px;'
                'position:relative;overflow:hidden;">'
                '<div style="position:absolute;left:50%;top:0;width:1px;height:100%;'
                'background:#777;"></div>'
                f'<div style="position:absolute;right:50%;top:0;width:{left_width:.1f}%;'
                f'height:100%;background:{color};border-radius:3px 0 0 3px;'
                'transition:width 0.08s;"></div>'
                f'<div style="position:absolute;left:50%;top:0;width:{right_width:.1f}%;'
                f'height:100%;background:{color};border-radius:0 3px 3px 0;'
                'transition:width 0.08s;"></div>'
                f'<span style="position:absolute;right:4px;top:0;line-height:18px;'
                f'font-size:0.72em;color:#fff;">{value_str}</span>'
                "</div></div>"
            )

        suffix = ""
        if len(means) > len(visible_names):
            suffix = (
                f'<div style="font-size:0.72em;color:#aaa;padding-top:3px;">'
                f"Showing {len(visible_names)} of {len(means)} scalars.</div>"
            )
        self._html_handle.content = (
            '<div style="padding:0.3em 0.5em;font-family:monospace;">'
            f'<div style="font-weight:600;margin:0 0 4px 0;">{html.escape(self._title)}</div>'
            + "".join(rows)
            + suffix
            + "</div>"
        )

    @staticmethod
    def _flatten(values: dict[str, Any]) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, list | tuple):
                if len(value) == 1:
                    result[name] = float(value[0])
                    continue
                for idx, item in enumerate(value):
                    result[f"{name}[{idx}]"] = float(item)
                continue
            result[name] = float(value)
        return result


class SimpleBodyVelocityTrackingPolicy:
    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        *,
        linear_gains: tuple[float, float, float] = (140.0, 140.0, 180.0),
        angular_gains: tuple[float, float, float] = (55.0, 55.0, 70.0),
    ):
        self._base_env = env.unwrapped
        self._linear_gains = torch.tensor(
            linear_gains,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 3)
        self._angular_gains = torch.tensor(
            angular_gains,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 3)
        taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)
        self._wrench_limits = torch.tensor(
            taluy_cfg.body_wrench_limit,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 6)

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        robot = self._base_env.scene["robot"]
        command = self._base_env.command_manager.get_command("body_velocity")
        assert command is not None
        lin_error = command[:, :3] - robot.data.root_link_lin_vel_b
        ang_error = command[:, 3:] - robot.data.root_link_ang_vel_b
        wrench = torch.cat(
            (lin_error * self._linear_gains, ang_error * self._angular_gains),
            dim=1,
        )
        wrench = torch.clamp(wrench, min=-self._wrench_limits, max=self._wrench_limits)
        return wrench / self._wrench_limits


class RollInspector:
    def __init__(
        self,
        base_env: ManagerBasedRlEnv,
        *,
        env_idx: int,
        print_period_s: float,
        panel_period_s: float,
        jsonl_path: Path | None,
    ) -> None:
        self._base_env = base_env
        self._env_idx = env_idx
        self._print_period_s = max(float(print_period_s), 0.0)
        self._panel_period_s = max(float(panel_period_s), 0.0)
        self._last_console_wall_time = 0.0
        self._last_panel_wall_time = 0.0
        self._last_wall_time = time.time()
        self._last_rate_wall_time = self._last_wall_time
        self._last_rate_step = int(base_env.common_step_counter)
        self._measured_steps_per_wall_s = 0.0
        self._jsonl_path = jsonl_path
        self._details_html_handle: Any | None = None
        self._actor_obs_panel: ScalarBarPanel | None = None
        self._critic_obs_panel: ScalarBarPanel | None = None
        self._viewer_frame_rate_hz: float | None = None
        self._odom_ref_pos_w: list[float] | None = None
        self._odom_ref_pitch_rad: float | None = None
        self._odom_ref_yaw_rad: float | None = None
        self._last_episode_step: int | None = None

        if self._jsonl_path is not None:
            self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def jsonl_path(self) -> Path | None:
        return self._jsonl_path

    def set_viewer_frame_rate(self, frame_rate_hz: float) -> None:
        self._viewer_frame_rate_hz = float(frame_rate_hz)

    def create_viser_gui(self, server: Any) -> None:
        with server.gui.add_folder("Roll Inspector"):
            self._actor_obs_panel = ScalarBarPanel(
                server,
                "Actor Observations",
                update_dt_s=max(self._panel_period_s, 1.0e-6),
            )
            self._critic_obs_panel = ScalarBarPanel(
                server,
                "Critic Observations",
                update_dt_s=max(self._panel_period_s, 1.0e-6),
            )
            self._details_html_handle = server.gui.add_html(
                '<div style="padding:0.5em;color:#999;font-size:0.85em;">'
                "Waiting for inspector data...</div>"
            )

    def maybe_emit(self, actions: torch.Tensor, *, force: bool = False) -> None:
        now = time.time()
        self._update_measured_rates(now)

        console_due = (
            self._print_period_s > 0.0
            and now - self._last_console_wall_time >= self._print_period_s
        )
        panel_due = (
            self._panel_period_s > 0.0
            and now - self._last_panel_wall_time >= self._panel_period_s
        )
        if not force and not console_due and not panel_due:
            return

        self._last_wall_time = now
        snapshot = self.snapshot(actions)
        if force or console_due:
            self._last_console_wall_time = now
            self._print_snapshot(snapshot)
            self._write_jsonl(snapshot)
        if force or panel_due:
            self._last_panel_wall_time = now
            self._update_panel(snapshot)

    def _update_measured_rates(self, now: float) -> None:
        dt = now - self._last_rate_wall_time
        if dt < 0.5:
            return
        step = int(self._base_env.common_step_counter)
        self._measured_steps_per_wall_s = (step - self._last_rate_step) / dt
        self._last_rate_wall_time = now
        self._last_rate_step = step

    def snapshot(self, actions: torch.Tensor) -> dict[str, Any]:
        env_idx = self._env_idx
        robot = self._base_env.scene["robot"]
        command = self._base_env.command_manager.get_command("body_velocity")
        if command is None:
            raise RuntimeError("body_velocity command is not available.")

        wrench_term = self._base_env.action_manager.get_term("body_wrench")
        if not isinstance(wrench_term, BodyWrenchAction):
            raise RuntimeError("body_wrench action term is not available.")

        _root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
        roll, pitch, yaw = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
        state = get_roll_task_state(self._base_env)
        policy_wrench = wrench_term.action_to_wrench(actions)
        current_pos_w = _root_pos_w[env_idx].detach().cpu().tolist()
        current_rpy_rad = [
            float(roll[env_idx].item()),
            float(pitch[env_idx].item()),
            float(yaw[env_idx].item()),
        ]
        current_episode_step = int(self._base_env.episode_length_buf[env_idx].item())
        if (
            self._odom_ref_pos_w is None
            or self._odom_ref_pitch_rad is None
            or self._odom_ref_yaw_rad is None
            or self._last_episode_step is None
            or current_episode_step < self._last_episode_step
        ):
            self._odom_ref_pos_w = list(current_pos_w)
            self._odom_ref_pitch_rad = current_rpy_rad[1]
            self._odom_ref_yaw_rad = current_rpy_rad[2]
        self._last_episode_step = current_episode_step

        position_from_start_w_m = [
            current_pos_w[i] - self._odom_ref_pos_w[i] for i in range(3)
        ]
        rpy_from_start_rad = [
            float(state.phi_total_rad[env_idx].item()),
            _wrap_angle_rad(current_rpy_rad[1] - self._odom_ref_pitch_rad),
            _wrap_angle_rad(current_rpy_rad[2] - self._odom_ref_yaw_rad),
        ]
        rpy_from_start_deg = [value * 180.0 / math.pi for value in rpy_from_start_rad]
        current_rpy_deg = [value * 180.0 / math.pi for value in current_rpy_rad]

        obs_terms = self._base_env.observation_manager.get_active_iterable_terms(env_idx)
        actor_obs = _iterable_terms_to_dict(obs_terms, prefix="actor-")
        critic_obs = _iterable_terms_to_dict(obs_terms, prefix="critic-")
        rewards = _iterable_terms_to_dict(
            self._base_env.reward_manager.get_active_iterable_terms(env_idx)
        )
        terminations = _iterable_terms_to_dict(
            self._base_env.termination_manager.get_active_iterable_terms(env_idx)
        )
        physics_dt_s = float(self._base_env.cfg.sim.mujoco.timestep)
        control_dt_s = float(self._base_env.step_dt)
        measured_actual_rt = self._measured_steps_per_wall_s * control_dt_s

        return {
            "time_s": self._last_wall_time,
            "step": int(self._base_env.common_step_counter),
            "env_idx": env_idx,
            "rates": {
                "physics_dt_s": physics_dt_s,
                "physics_hz": 1.0 / physics_dt_s,
                "decimation": int(self._base_env.cfg.decimation),
                "control_dt_s": control_dt_s,
                "control_hz_sim": 1.0 / control_dt_s,
                "observation_hz_sim": 1.0 / control_dt_s,
                "measured_steps_per_wall_s": self._measured_steps_per_wall_s,
                "measured_observation_hz_wall": self._measured_steps_per_wall_s,
                "measured_actual_rt": measured_actual_rt,
                "viewer_frame_rate_hz": self._viewer_frame_rate_hz,
            },
            "command_b": command[env_idx].detach().cpu().tolist(),
            "action": actions[env_idx].detach().cpu().tolist(),
            "policy_wrench_b": policy_wrench[env_idx].detach().cpu().tolist(),
            "last_applied_wrench_b": wrench_term.desired_wrench_b[env_idx]
            .detach()
            .cpu()
            .tolist(),
            "last_applied_wrench_origin_b": wrench_term.applied_wrench_origin_b[env_idx]
            .detach()
            .cpu()
            .tolist(),
            "thruster_targets_n": wrench_term.thruster_targets[env_idx]
            .detach()
            .cpu()
            .tolist(),
            "thruster_max_abs_n": float(
                wrench_term.thruster_targets[env_idx].abs().max().item()
            ),
            "thruster_saturation_fraction": float(
                wrench_term.step_saturation_fraction[env_idx].item()
            ),
            "lin_vel_b": robot.data.root_link_lin_vel_b[env_idx].detach().cpu().tolist(),
            "ang_vel_b": robot.data.root_link_ang_vel_b[env_idx].detach().cpu().tolist(),
            "odometry": {
                "pose.position_w_m": current_pos_w,
                "pose.position_from_start_w_m": position_from_start_w_m,
                "pose.rpy_current_rad": current_rpy_rad,
                "pose.rpy_current_deg": current_rpy_deg,
                "pose.rpy_from_start_rad": rpy_from_start_rad,
                "pose.rpy_from_start_deg": rpy_from_start_deg,
                "twist.linear_b_m_s": robot.data.root_link_lin_vel_b[env_idx]
                .detach()
                .cpu()
                .tolist(),
                "twist.angular_b_rad_s": robot.data.root_link_ang_vel_b[env_idx]
                .detach()
                .cpu()
                .tolist(),
                "twist.angular_b_deg_s": [
                    float(value) * 180.0 / math.pi
                    for value in robot.data.root_link_ang_vel_b[env_idx]
                    .detach()
                    .cpu()
                    .tolist()
                ],
            },
            "quat_wxyz": quat_wxyz[env_idx].detach().cpu().tolist(),
            "euler_rpy_rad": current_rpy_rad,
            "roll_state": {
                "phi_total_rad": _tensor_env_value(state.phi_total_rad, env_idx),
                "delta_roll_rad": _tensor_env_value(state.delta_roll_rad, env_idx),
                "z_ref_m": _tensor_env_value(state.z_ref_m, env_idx),
                "psi_ref_rad": _tensor_env_value(state.psi_ref_rad, env_idx),
                "xy_ref_w": _tensor_env_value(state.xy_ref_w, env_idx),
                "target_reached": _tensor_env_value(state.target_reached, env_idx),
                "settle_counter_steps": _tensor_env_value(
                    state.settle_counter_steps, env_idx
                ),
            },
            "actor_observations": actor_obs,
            "critic_observations": critic_obs,
            "last_step_rewards": rewards,
            "terminations": terminations,
        }

    def _print_snapshot(self, snapshot: dict[str, Any]) -> None:
        rates = snapshot["rates"]
        print(
            "[roll-inspector] "
            f"step={snapshot['step']} env={snapshot['env_idx']} "
            f"rt={rates['measured_actual_rt']:.3f} "
            f"obs_wall_hz={rates['measured_observation_hz_wall']:.1f} "
            f"command_b={snapshot['command_b']} "
            f"rpy={snapshot['euler_rpy_rad']} "
            f"action={snapshot['action']} "
            f"policy_wrench_b={snapshot['policy_wrench_b']} "
            f"last_sat_frac={snapshot['thruster_saturation_fraction']:.3f} "
            f"phi_total_rad={snapshot['roll_state']['phi_total_rad']:.5f} "
            f"rewards={snapshot['last_step_rewards']} "
            f"terminations={snapshot['terminations']}"
        )

    def _write_jsonl(self, snapshot: dict[str, Any]) -> None:
        if self._jsonl_path is None:
            return
        with self._jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(snapshot, separators=(",", ":")) + "\n")

    def _update_panel(self, snapshot: dict[str, Any]) -> None:
        if self._details_html_handle is None:
            return

        if self._actor_obs_panel is not None:
            self._actor_obs_panel.update(snapshot["actor_observations"])
        if self._critic_obs_panel is not None:
            self._critic_obs_panel.update(snapshot["critic_observations"])

        command_action = {
            "command_b": snapshot["command_b"],
            "lin_vel_b": snapshot["lin_vel_b"],
            "ang_vel_b": snapshot["ang_vel_b"],
            "action": snapshot["action"],
            "policy_wrench_b": snapshot["policy_wrench_b"],
            "last_applied_wrench_b": snapshot["last_applied_wrench_b"],
            "thruster_max_abs_n": snapshot["thruster_max_abs_n"],
            "thruster_saturation_fraction": snapshot["thruster_saturation_fraction"],
        }
        rates = snapshot["rates"]
        rate_values = {
            "physics_hz": rates["physics_hz"],
            "control_hz_sim": rates["control_hz_sim"],
            "observation_hz_sim": rates["observation_hz_sim"],
            "measured_observation_hz_wall": rates["measured_observation_hz_wall"],
            "measured_actual_rt": rates["measured_actual_rt"],
            "viewer_frame_rate_hz": rates["viewer_frame_rate_hz"],
        }
        sections = [
            self._html_kv_section("Rates", rate_values),
            self._html_kv_section("Odometry", snapshot["odometry"]),
            self._html_kv_section("Command / Action", command_action),
            self._html_kv_section("Roll State", snapshot["roll_state"]),
            self._html_kv_section("Last Step Rewards", snapshot["last_step_rewards"]),
            self._html_kv_section("Terminations", snapshot["terminations"]),
        ]
        self._details_html_handle.content = (
            '<div style="padding:0.45em 0.55em;font-family:monospace;'
            'background:rgba(20,24,31,0.92);color:#f3f4f6;border:1px solid #334155;'
            'border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,0.18);">'
            f'<div style="font-size:1.0em;font-weight:700;margin-bottom:4px;color:#f8fafc;">'
            f"Env {snapshot['env_idx']} | Step {snapshot['step']}</div>"
            f'<div style="font-size:0.80em;margin-bottom:8px;color:#cbd5e1;'
            'padding:6px 8px;background:rgba(51,65,85,0.45);border-radius:4px;">'
            f"Odometry pose xyz from start [m]: "
            f"{html.escape(_compact_value(snapshot['odometry']['pose.position_from_start_w_m']))}<br>"
            f"Odometry pose rpy from start [deg]: "
            f"{html.escape(_compact_value(snapshot['odometry']['pose.rpy_from_start_deg']))}<br>"
            f"Twist linear [m/s]: "
            f"{html.escape(_compact_value(snapshot['odometry']['twist.linear_b_m_s']))}<br>"
            f"Twist angular [deg/s]: "
            f"{html.escape(_compact_value(snapshot['odometry']['twist.angular_b_deg_s']))}"
            "</div>"
            + "".join(sections)
            + "</div>"
        )

    @staticmethod
    def _html_kv_section(title: str, values: dict[str, Any]) -> str:
        rows = []
        for key, value in values.items():
            safe_key = html.escape(key, quote=True)
            safe_value = html.escape(_compact_value(value), quote=True)
            rows.append(
                '<div style="display:flex;align-items:flex-start;gap:6px;'
                'margin:2px 0;font-size:0.76em;line-height:1.35;">'
                f'<span style="min-width:132px;text-align:right;color:#cbd5e1;'
                f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;" '
                f'title="{safe_key}">{safe_key}</span>'
                f'<span style="color:#f8fafc;word-break:break-word;flex:1;">{safe_value}</span>'
                "</div>"
            )
        return (
            '<div style="margin-top:8px;padding-top:6px;border-top:1px solid #334155;">'
            f'<div style="font-weight:700;margin-bottom:4px;color:#f8fafc;">'
            f"{html.escape(title)}</div>"
            + "".join(rows)
            + "</div>"
        )


class InspectablePolicy:
    def __init__(
        self,
        base_policy: Callable[[object], torch.Tensor],
        inspector: RollInspector,
    ) -> None:
        self._base_policy = base_policy
        self._inspector = inspector

    def __call__(self, obs: object) -> torch.Tensor:
        actions = self._base_policy(obs)
        self._inspector.maybe_emit(actions)
        return actions


class RollInspectorViserPlayViewer(ViserPlayViewer):
    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        policy: Callable[[Any], torch.Tensor],
        inspector: RollInspector,
        *,
        viser_server: Any,
    ) -> None:
        super().__init__(
            env,
            cast(Any, policy),
            viser_server=viser_server,
        )
        self._roll_inspector = inspector
        self._roll_inspector.set_viewer_frame_rate(self.frame_rate)

    def setup(self) -> None:
        super().setup()
        self._roll_inspector.create_viser_gui(self._server)

    def _execute_step(self) -> bool:
        try:
            with torch.no_grad():
                obs = self.env.get_observations()
                actions = self.policy(obs)
                self.env.step(actions)
                self._step_count += 1
                self._stats_steps += 1
                self._roll_inspector.maybe_emit(actions)
                return True
        except Exception:
            self._last_error = traceback.format_exc()
            self.log(
                f"[ERROR] Exception during step:\n{self._last_error}",
                VerbosityLevel.SILENT,
            )
            self.pause()
            return False


def _set_fixed_command(base_env: ManagerBasedRlEnv, values: tuple[float, ...]) -> None:
    if len(values) != 6:
        raise ValueError(f"Expected 6 command values, got {len(values)}.")

    term = base_env.command_manager.get_term("body_velocity")
    if not isinstance(term, UniformBodyVelocityCommand):
        raise RuntimeError("Taluy body_velocity command term is not available.")

    command = torch.tensor(values, device=base_env.device, dtype=torch.float).view(1, 6)
    term.vel_command_b[:] = command.expand(base_env.num_envs, -1)
    term.is_zero_env[:] = False
    term.time_left[:] = 1.0e9


def _run_dry_steps(
    env: RslRlVecEnvWrapper,
    policy: Callable[[object], torch.Tensor],
    inspector: RollInspector,
    *,
    num_steps: int,
    fixed_command: tuple[float, ...],
) -> None:
    base_env = env.unwrapped
    env.reset()
    _set_fixed_command(base_env, fixed_command)

    reward = torch.zeros(env.num_envs, device=base_env.device)
    for _ in range(num_steps):
        obs = env.get_observations()
        actions = policy(obs)
        _, reward, _, _ = env.step(actions)
        inspector.maybe_emit(actions, force=True)

    command = base_env.command_manager.get_command("body_velocity")
    assert command is not None
    snapshot = inspector.snapshot(actions)
    print("Dry-run complete.")
    print(f"  command_b={command.detach().cpu()[0].tolist()}")
    print(f"  reward={reward.detach().cpu().tolist()}")
    print(f"  final_rpy_rad={snapshot['euler_rpy_rad']}")
    if inspector.jsonl_path is not None:
        print(f"  jsonl={inspector.jsonl_path}")


def _run_viser_viewer(
    env: RslRlVecEnvWrapper,
    policy: Callable[[Any], torch.Tensor],
    inspector: RollInspector,
    host: str,
    port: int,
) -> None:
    server = _build_viser_server(host, port)
    RollInspectorViserPlayViewer(
        env,
        policy,
        inspector,
        viser_server=server,
    ).run()


def _print_debug_legend() -> None:
    print(
        "Debug legend: purple = commanded linear velocity, blue = measured "
        "linear velocity; both share one anchor above the robot center."
    )
    print(
        "             orange/pink thruster arrows = commanded thrust along each "
        "thruster's local -Z force axis; sub-1 N commands are hidden."
    )


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    viewer = _resolve_viewer(args.viewer)

    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.dry_run_steps < 0:
        raise SystemExit("--dry-run-steps must be non-negative.")
    if not 0 <= args.inspect_env_idx < args.num_envs:
        raise SystemExit("--inspect-env-idx must satisfy 0 <= idx < --num-envs.")

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    jsonl_path = None if args.no_jsonl else args.log_jsonl or _default_jsonl_path()
    base_env = ManagerBasedRlEnv(
        cfg=_make_roll_inspector_env_cfg(args),
        device=device,
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=1.0)
    policy = SimpleBodyVelocityTrackingPolicy(env)
    inspector = RollInspector(
        base_env,
        env_idx=args.inspect_env_idx,
        print_period_s=args.print_period_s,
        panel_period_s=args.panel_period_s,
        jsonl_path=jsonl_path,
    )

    fixed_command = (
        tuple(float(value) for value in args.fixed_command)
        if args.fixed_command is not None
        else DEFAULT_COMMAND
    )

    try:
        if args.dry_run_steps > 0:
            _run_dry_steps(
                env,
                policy,
                inspector,
                num_steps=args.dry_run_steps,
                fixed_command=fixed_command,
            )
            return

        env.reset()
        if args.fixed_command is not None:
            _set_fixed_command(base_env, fixed_command)

        print(f"Taluy roll interactive inspector | device={device} viewer={viewer}")
        print("Policy: hand-tuned body-velocity feedback -> body_wrench action")
        if jsonl_path is not None:
            print(f"JSONL telemetry: {jsonl_path}")
        _print_debug_legend()
        if viewer == "viser":
            print(
                "Use `Commands -> Body Velocity` to set the 6D reference, "
                "`Scene -> Body_velocity` for overlays, and `Roll Inspector` "
                "for live training telemetry."
            )
        else:
            print("Native viewer has no Viser panel; console/JSONL telemetry remains active.")

        if viewer == "native":
            NativeMujocoViewer(env, InspectablePolicy(policy, inspector)).run()
        else:
            _run_viser_viewer(env, policy, inspector, args.viser_host, args.viser_port)
    finally:
        env.close()


if __name__ == "__main__":
    main()
