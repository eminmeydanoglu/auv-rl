"""Replay a recorded model rollout in Viser at a chosen real-time factor."""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path
import socket
import time
from typing import Any

import numpy as np

from ._model_replay_io import done_reason_name, load_recording


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording_dir", type=Path)
    parser.add_argument("--viewer", choices=("viser", "none"), default="viser")
    parser.add_argument("--rtf", type=float, default=1.0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--start-paused", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--viser-host", default="0.0.0.0")
    parser.add_argument("--viser-port", type=int, default=0)
    parser.add_argument(
        "--dry-run-frames",
        type=int,
        default=0,
        help="Validate replay data without opening Viser.",
    )
    return parser.parse_args()


def _pick_unused_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _build_viser_server(host: str, port: int):
    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    if port == 0:
        port = _pick_unused_local_port()
    try:
        return viser.ViserServer(host=host, port=port, label="model-replay")
    except TypeError:
        print("Warning: this viser version does not support host/port arguments.")
        return viser.ViserServer(label="model-replay")


def _server_url(server: Any) -> str:
    host = str(server.get_host())
    port = int(server.get_port())
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    return f"http://{display_host}:{port}"


def _episode_indices(arrays: dict[str, np.ndarray], episode: int) -> np.ndarray:
    if "episode_idx" not in arrays:
        return np.arange(arrays["time_s"].shape[0])
    indices = np.nonzero(arrays["episode_idx"] == episode)[0]
    if indices.size == 0:
        raise ValueError(f"Episode {episode} has no frames in this recording.")
    return indices


def _compact(value: Any) -> str:
    if isinstance(value, float | np.floating):
        return f"{float(value):.5g}"
    if isinstance(value, int | bool | np.integer | np.bool_):
        return str(value)
    if isinstance(value, np.ndarray):
        return _compact(value.tolist())
    if isinstance(value, list | tuple):
        return "[" + ", ".join(_compact(item) for item in value) + "]"
    return str(value)


def _html_table(title: str, values: dict[str, Any]) -> str:
    rows = []
    for key, value in values.items():
        rows.append(
            '<div style="display:flex;gap:8px;margin:2px 0;font-size:0.78em;">'
            f'<span style="min-width:142px;text-align:right;color:#cbd5e1;">'
            f"{html.escape(key)}</span>"
            f'<span style="color:#f8fafc;word-break:break-word;">'
            f"{html.escape(_compact(value))}</span></div>"
        )
    return (
        '<div style="margin-top:8px;padding-top:6px;border-top:1px solid #334155;">'
        f'<div style="font-weight:700;color:#f8fafc;margin-bottom:4px;">'
        f"{html.escape(title)}</div>"
        + "".join(rows)
        + "</div>"
    )


class ReplayTimeline:
    def __init__(
        self,
        frame_indices: np.ndarray,
        time_s: np.ndarray,
        *,
        rtf: float,
        frame_stride: int,
        start_paused: bool,
        loop: bool,
    ) -> None:
        if rtf <= 0.0:
            raise ValueError("--rtf must be positive.")
        if frame_stride <= 0:
            raise ValueError("--frame-stride must be positive.")
        self.frame_indices = frame_indices[::frame_stride]
        self.time_s = time_s[self.frame_indices]
        self.rtf = float(rtf)
        self.loop = loop
        self.paused = bool(start_paused)
        self.local_cursor = 0
        self._wall_anchor = time.time()
        self._sim_anchor = float(self.time_s[0])

    @property
    def current_frame(self) -> int:
        return int(self.frame_indices[self.local_cursor])

    def set_local_cursor(self, cursor: int) -> None:
        self.local_cursor = int(np.clip(cursor, 0, len(self.frame_indices) - 1))
        self._wall_anchor = time.time()
        self._sim_anchor = float(self.time_s[self.local_cursor])

    def toggle_pause(self) -> None:
        if not self.paused:
            self.set_local_cursor(self.local_cursor)
        self.paused = not self.paused

    def advance(self) -> bool:
        if self.paused:
            return True
        target_sim_time = self._sim_anchor + (time.time() - self._wall_anchor) * self.rtf
        next_cursor = int(np.searchsorted(self.time_s, target_sim_time, side="right") - 1)
        if next_cursor >= len(self.frame_indices):
            if not self.loop:
                self.local_cursor = len(self.frame_indices) - 1
                self.paused = True
                return False
            self.local_cursor = 0
            self._wall_anchor = time.time()
            self._sim_anchor = float(self.time_s[0])
            return True
        self.local_cursor = max(next_cursor, 0)
        return True


class ViserReplay:
    def __init__(
        self,
        arrays: dict[str, np.ndarray],
        manifest: dict[str, Any],
        timeline: ReplayTimeline,
        server: Any,
    ) -> None:
        self.arrays = arrays
        self.manifest = manifest
        self.timeline = timeline
        self.server = server
        self._updating_slider = False

        server.scene.add_grid(
            "/grid",
            width=8.0,
            height=8.0,
            width_segments=16,
            height_segments=16,
            position=(0.0, 0.0, -0.5),
        )
        self.body = server.scene.add_box(
            "/vehicle/body",
            dimensions=(0.7, 0.35, 0.22),
            color=(36, 111, 255),
            opacity=0.62,
        )
        self.frame = server.scene.add_frame(
            "/vehicle/frame",
            axes_length=0.42,
            axes_radius=0.014,
            origin_radius=0.035,
        )
        self.trail = server.scene.add_line_segments(
            "/vehicle/trail",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(80, 220, 160),
            line_width=2.0,
        )
        self.wrench = server.scene.add_line_segments(
            "/vehicle/wrench",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(255, 170, 40),
            line_width=4.0,
        )
        self.thrusters = server.scene.add_line_segments(
            "/vehicle/thrusters",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(255, 80, 160),
            line_width=2.0,
        )
        self.info = server.gui.add_html("")

        with server.gui.add_folder("Replay"):
            self.play_button = server.gui.add_button("Pause" if not timeline.paused else "Play")
            self.slider = server.gui.add_slider(
                "Frame",
                min=0,
                max=max(len(timeline.frame_indices) - 1, 0),
                step=1,
                initial_value=timeline.local_cursor,
            )
            self.rtf_number = server.gui.add_number(
                "RTF",
                initial_value=timeline.rtf,
                min=0.05,
                step=0.05,
            )
            self.jump_done = server.gui.add_button("Jump to first done")

        @self.play_button.on_click
        def _(_) -> None:
            self.timeline.toggle_pause()
            self.play_button.label = "Play" if self.timeline.paused else "Pause"

        @self.slider.on_update
        def _(event) -> None:
            if self._updating_slider:
                return
            self.timeline.set_local_cursor(int(event.target.value))
            self.update()

        @self.rtf_number.on_update
        def _(event) -> None:
            self.timeline.set_local_cursor(self.timeline.local_cursor)
            self.timeline.rtf = max(float(event.target.value), 0.05)

        @self.jump_done.on_click
        def _(_) -> None:
            if "would_done" not in self.arrays:
                return
            done_indices = [
                idx
                for idx, frame_idx in enumerate(self.timeline.frame_indices)
                if bool(self.arrays["would_done"][frame_idx])
            ]
            if done_indices:
                self.timeline.set_local_cursor(done_indices[0])
                self.update()

    def update(self) -> None:
        frame_idx = self.timeline.current_frame
        root_pos = self.arrays["root_pos_w"][frame_idx].astype(float)
        quat = self.arrays["quat_wxyz"][frame_idx].astype(float)
        self.body.position = root_pos
        self.body.wxyz = quat
        self.frame.position = root_pos
        self.frame.wxyz = quat

        episode = int(self.arrays.get("episode_idx", np.asarray([0]))[frame_idx])
        ep_mask = self.arrays.get("episode_idx", np.zeros_like(self.arrays["time_s"])) == episode
        ep_indices = np.nonzero(ep_mask)[0]
        ep_indices = ep_indices[ep_indices <= frame_idx]
        if ep_indices.size > 1:
            pts = self.arrays["root_pos_w"][ep_indices].astype(np.float32)
            self.trail.points = np.stack([pts[:-1], pts[1:]], axis=1)

        if "policy_wrench_b" in self.arrays:
            wrench = self.arrays["policy_wrench_b"][frame_idx].astype(float)
            force = np.asarray(wrench[:3], dtype=np.float32)
            norm = float(np.linalg.norm(force))
            if norm > 1.0e-6:
                end = root_pos + force / max(norm, 1.0) * min(norm / 180.0, 0.75)
                self.wrench.points = np.asarray([[root_pos, end]], dtype=np.float32)

        if "thruster_origin_w" in self.arrays and "thruster_force_axis_w" in self.arrays:
            origins = self.arrays["thruster_origin_w"][frame_idx].astype(np.float32)
            axes = self.arrays["thruster_force_axis_w"][frame_idx].astype(np.float32)
            targets = self.arrays.get("thruster_targets_n")
            if targets is not None:
                mag = targets[frame_idx].astype(np.float32).reshape(-1, 1)
                ends = origins + axes * np.clip(mag / 60.0, -1.0, 1.0) * 0.35
                self.thrusters.points = np.stack([origins, ends], axis=1)

        reward_names = self.manifest.get("reward_term_names", [])
        rewards = {}
        if "reward_terms" in self.arrays:
            for name, value in zip(reward_names, self.arrays["reward_terms"][frame_idx], strict=False):
                rewards[str(name)] = float(value)

        rpy = self.arrays["rpy_rad"][frame_idx]
        reason_code = int(self.arrays.get("would_done_reason_code", np.asarray([0]))[frame_idx])
        state_values = {
            "frame": f"{self.timeline.local_cursor}/{len(self.timeline.frame_indices) - 1}",
            "episode": episode,
            "time_s": float(self.arrays["time_s"][frame_idx]),
            "done_reason": done_reason_name(reason_code),
            "controller_stopped": bool(self.arrays.get("controller_stopped", np.asarray([False]))[frame_idx]),
            "roll_deg": math.degrees(float(rpy[0])),
            "pitch_deg": math.degrees(float(rpy[1])),
            "yaw_deg": math.degrees(float(rpy[2])),
            "phi_total_rad": float(self.arrays.get("phi_total_rad", np.asarray([0.0]))[frame_idx]),
            "roll_progress_ratio": float(self.arrays.get("roll_progress_ratio", np.asarray([0.0]))[frame_idx]),
            "xy_drift_m": float(self.arrays.get("xy_drift_m", np.asarray([0.0]))[frame_idx]),
            "depth_error_m": float(self.arrays.get("depth_error_m", np.asarray([0.0]))[frame_idx]),
            "thruster_saturation": float(
                self.arrays.get("thruster_saturation_fraction", np.asarray([0.0]))[frame_idx]
            ),
        }
        self.info.content = (
            '<div style="padding:0.55em;font-family:monospace;'
            'background:rgba(15,23,42,0.94);border:1px solid #334155;'
            'border-radius:6px;color:#f8fafc;">'
            f'<div style="font-weight:800;font-size:1.0em;">'
            f"{html.escape(str(self.manifest.get('curriculum_stage') or 'model replay'))}"
            "</div>"
            + _html_table("State", state_values)
            + _html_table("Reward Terms", rewards)
            + "</div>"
        )
        if int(self.slider.value) != int(self.timeline.local_cursor):
            self._updating_slider = True
            try:
                self.slider.value = self.timeline.local_cursor
            finally:
                self._updating_slider = False


def _dry_run(arrays: dict[str, np.ndarray], frame_indices: np.ndarray, count: int) -> None:
    for local_idx, frame_idx in enumerate(frame_indices[:count]):
        print(
            f"frame={local_idx} raw={int(frame_idx)} "
            f"time_s={float(arrays['time_s'][frame_idx]):.3f} "
            f"pos={arrays['root_pos_w'][frame_idx].tolist()} "
            f"rpy={arrays['rpy_rad'][frame_idx].tolist()} "
            f"done={bool(arrays.get('would_done', np.asarray([False]))[frame_idx])}"
        )


def main() -> None:
    args = _parse_args()
    if args.rtf <= 0.0:
        raise SystemExit("--rtf must be positive.")
    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive.")

    recording = load_recording(args.recording_dir)
    frame_indices = _episode_indices(recording.arrays, args.episode)
    timeline = ReplayTimeline(
        frame_indices,
        recording.arrays["time_s"],
        rtf=args.rtf,
        frame_stride=args.frame_stride,
        start_paused=args.start_paused,
        loop=args.loop,
    )

    if args.dry_run_frames > 0 or args.viewer == "none":
        count = args.dry_run_frames or min(10, len(timeline.frame_indices))
        _dry_run(recording.arrays, timeline.frame_indices, count)
        return

    server = _build_viser_server(args.viser_host, args.viser_port)
    print(f"Model replay ready: {_server_url(server)}")
    print(f"Recording: {recording.root}")
    replay = ViserReplay(recording.arrays, recording.manifest, timeline, server)
    replay.update()
    try:
        while True:
            timeline.advance()
            replay.update()
            time.sleep(1.0 / 60.0)
    except KeyboardInterrupt:
        server.stop()


if __name__ == "__main__":
    main()
