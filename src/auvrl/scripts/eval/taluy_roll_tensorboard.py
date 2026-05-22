"""Evaluate Taluy roll policies and publish an easy TensorBoard dashboard.

The script accepts either checkpoints, model replay recordings, or older
``timeseries_raw.npz`` reports. It writes a compact TensorBoard run plus
JSON/PNG artifacts under ``logs/tensorboard_eval/roll/<suite>/<label>/``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import glob
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = ROOT / "logs" / "tensorboard_eval" / "roll"

TRAJECTORY_METRICS = (
    "roll_progress_ratio",
    "pitch_abs_rad",
    "yaw_abs_error_rad",
    "xy_drift_m",
    "depth_abs_error_m",
    "body_wrench_saturation_fraction",
    "body_wrench_action_l2",
    "root_ang_speed_rad_s",
)

# Extra per-step trajectories captured from the live env. These are written
# into TensorBoard as ``EvalTrajectoryExtraMean/<metric>`` etc., but are not
# part of the curated TERMINAL_METRICS snapshot.
SIGNED_RPY_METRICS = ("roll_rad", "pitch_rad", "yaw_rad")
DRIFT_COMPONENT_METRICS = ("x_drift_m", "y_drift_m", "z_drift_m")
LIN_VEL_METRICS = ("lin_vel_x_b", "lin_vel_y_b", "lin_vel_z_b")
ANG_VEL_METRICS = ("ang_vel_x_b", "ang_vel_y_b", "ang_vel_z_b")
WRENCH_COMPONENT_METRICS = (
    "wrench_Fx",
    "wrench_Fy",
    "wrench_Fz",
    "wrench_Tx",
    "wrench_Ty",
    "wrench_Tz",
)
ACTION_METRICS = ("action_rate_l2", "thruster_max_abs_n", "hydro_wrench_norm")
EXTENDED_TRAJECTORY_METRICS = (
    SIGNED_RPY_METRICS
    + DRIFT_COMPONENT_METRICS
    + LIN_VEL_METRICS
    + ANG_VEL_METRICS
    + WRENCH_COMPONENT_METRICS
    + ACTION_METRICS
)

TERMINAL_METRICS = (
    "roll_progress_ratio",
    "phi_total_rad",
    "pitch_abs_rad",
    "yaw_abs_error_rad",
    "xy_drift_m",
    "depth_abs_error_m",
    "xy_drift_m_peak",
    "body_wrench_saturation_fraction",
    "body_wrench_action_l2",
    "root_ang_speed_rad_s",
    "thruster_max_abs_n",
    "action_rate_l2",
)

OUTCOME_REASON_NAMES = (
    "task_success",
    "excess_pitch",
    "excess_depth_error",
    "excess_xy_drift",
    "time_out",
    "multiple",
)


@dataclass(frozen=True)
class EvalSeries:
    """Vectorized rollout metrics with shape ``(time, env)`` for each metric.

    Per-thruster commanded forces and per-reward-term contributions are stored
    in dedicated 3-D arrays of shape ``(time, env, channel)`` because the
    channel count depends on the policy / env wiring at evaluation time.
    """

    label: str
    source: str
    arrays: dict[str, np.ndarray]
    time_s: np.ndarray
    first_done_step: np.ndarray
    first_done_time_s: np.ndarray
    done_reason: np.ndarray
    metadata: dict[str, Any]
    thruster_targets_n: np.ndarray | None = None  # (time, env, n_thr)
    thruster_names: tuple[str, ...] = ()
    thruster_saturated: np.ndarray | None = None  # (time, env, n_thr) bool/float
    reward_terms_step: np.ndarray | None = None  # (time, env, n_terms)
    reward_term_names: tuple[str, ...] = ()
    action_dims: np.ndarray | None = None  # (time, env, action_dim) clipped policy action
    action_dim_names: tuple[str, ...] = ()

    @property
    def num_steps(self) -> int:
        return int(self.time_s.shape[0])

    @property
    def num_envs(self) -> int:
        if self.first_done_step.size:
            return int(self.first_done_step.shape[0])
        for values in self.arrays.values():
            if values.ndim == 2:
                return int(values.shape[1])
        return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[])
    parser.add_argument(
        "--checkpoint-glob",
        action="append",
        default=[],
        help="Glob for checkpoint files. Example: 'logs/.../model_*.pt'.",
    )
    parser.add_argument(
        "--recording-dir",
        type=Path,
        action="append",
        default=[],
        help="Directory produced by auvrl.scripts.demo.model_record.",
    )
    parser.add_argument(
        "--timeseries-npz",
        type=Path,
        action="append",
        default=[],
        help="Older roll time-series raw npz file.",
    )
    parser.add_argument("--suite", default=None)
    parser.add_argument("--label", default=None, help="Label for a single input.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--curriculum-stage", default=None)
    parser.add_argument("--episode-length-s", type=float, default=None)
    parser.add_argument("--roll-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Write TensorBoard/JSON only; skip PNG summary generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into an existing non-empty run directory.",
    )
    return parser.parse_args()


def _safe_label(value: str) -> str:
    value = value.strip().replace(os.sep, "_")
    value = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)
    value = value.strip("._")
    return value or "eval"


def _checkpoint_step(path: Path) -> int:
    match = re.search(r"model_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else 0


def _checkpoint_label(path: Path) -> str:
    parent = path.parent.name
    suffix = path.stem
    return _safe_label(f"{parent}_{suffix}")


def _resolve_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths = [path.expanduser() for path in args.checkpoint]
    for pattern in args.checkpoint_glob:
        paths.extend(Path(match) for match in glob.glob(pattern))
    unique = sorted({path.resolve() for path in paths}, key=lambda item: (str(item.parent), _checkpoint_step(item)))
    for path in unique:
        if not path.exists():
            raise SystemExit(f"Checkpoint not found: {path}")
    return unique


def _make_suite_name(value: str | None) -> str:
    if value:
        return _safe_label(value)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{stamp}_roll_eval"


def _ensure_output_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(
            f"Output directory already exists and is not empty: {path}\n"
            "Use --overwrite or choose a different --suite/--label."
        )
    path.mkdir(parents=True, exist_ok=True)


def _to_2d(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"{name} must be 1D or 2D, got shape {array.shape}.")


def _nan_stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": math.nan, "p10": math.nan, "p50": math.nan, "p90": math.nan, "max": math.nan}
    return {
        "mean": float(np.mean(finite)),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
        "max": float(np.max(finite)),
    }


def _take_first_done(values: np.ndarray, first_done_step: np.ndarray) -> np.ndarray:
    values_2d = _to_2d(values, name="values")
    out = np.full(values_2d.shape[1], np.nan, dtype=np.float64)
    for env_idx, step in enumerate(first_done_step):
        if step < 0:
            continue
        row = min(int(step), values_2d.shape[0] - 1)
        out[env_idx] = float(values_2d[row, env_idx])
    return out


def _last_active(values: np.ndarray, first_done_step: np.ndarray) -> np.ndarray:
    values_2d = _to_2d(values, name="values")
    out = np.full(values_2d.shape[1], np.nan, dtype=np.float64)
    for env_idx, step in enumerate(first_done_step):
        if step >= 0:
            row = min(int(step), values_2d.shape[0] - 1)
        else:
            row = values_2d.shape[0] - 1
        out[env_idx] = float(values_2d[row, env_idx])
    return out


def _mask_after_done(values: np.ndarray, first_done_step: np.ndarray) -> np.ndarray:
    values_2d = _to_2d(values, name="values").astype(np.float64).copy()
    for env_idx, step in enumerate(first_done_step):
        if step >= 0:
            values_2d[int(step) + 1 :, env_idx] = np.nan
    return values_2d


def summarize_eval(series: EvalSeries) -> dict[str, Any]:
    done_mask = series.first_done_step >= 0
    summary: dict[str, Any] = {
        "label": series.label,
        "source": series.source,
        "num_steps": series.num_steps,
        "num_envs": series.num_envs,
        "done_count": int(np.sum(done_mask)),
        "done_rate": float(np.mean(done_mask)) if done_mask.size else math.nan,
        "first_done_time_s": _nan_stats(series.first_done_time_s[done_mask]),
        "metadata": series.metadata,
        "outcome": {},
        "terminal": {},
        "trajectory": {},
    }

    if series.done_reason.size:
        for code, name in enumerate(OUTCOME_REASON_NAMES, start=1):
            summary["outcome"][f"{name}_rate"] = float(np.mean(series.done_reason == code))
    if "would_success" in series.arrays:
        first_success = _take_first_done(series.arrays["would_success"], series.first_done_step)
        summary["outcome"]["success_rate"] = float(np.nanmean(first_success > 0.5))
    elif "target_reached" in series.arrays:
        first_reached = _take_first_done(series.arrays["target_reached"], series.first_done_step)
        summary["outcome"]["target_reached_rate"] = float(np.nanmean(first_reached > 0.5))

    for metric in TERMINAL_METRICS:
        if metric not in series.arrays:
            continue
        terminal = _last_active(series.arrays[metric], series.first_done_step)
        first_done = _take_first_done(series.arrays[metric], series.first_done_step)
        summary["terminal"][f"final_{metric}"] = _nan_stats(terminal)
        summary["terminal"][f"first_done_{metric}"] = _nan_stats(first_done)

    for metric in TRAJECTORY_METRICS:
        if metric not in series.arrays:
            continue
        values = _mask_after_done(series.arrays[metric], series.first_done_step)
        env_mean = np.nanmean(values, axis=1)
        summary["trajectory"][metric] = {
            "time_mean_of_env_mean": float(np.nanmean(env_mean)),
            "peak_env_mean": float(np.nanmax(env_mean)),
            "final_env_mean": float(env_mean[-1]),
        }
    return summary


def _load_timeseries_npz(path: Path, *, label: str | None = None) -> EvalSeries:
    data = np.load(path, allow_pickle=False)
    arrays = {
        name: _to_2d(data[name], name=name).astype(np.float32)
        for name in data.files
        if data[name].ndim in (1, 2) and name not in {"first_done_step", "first_done_time_s"}
    }
    first_done_step_raw = (
        np.asarray(data["first_done_step"], dtype=np.int64)
        if "first_done_step" in data.files
        else _derive_first_done_step(arrays)
    )
    default_time = _default_time(arrays)
    first_done_step = _normalize_report_steps(first_done_step_raw, time_len=default_time.shape[0])
    first_done_time_s = (
        np.asarray(data["first_done_time_s"], dtype=np.float64)
        if "first_done_time_s" in data.files
        else _step_to_time(first_done_step, default_time)
    )
    time_s = _infer_time_s(arrays, first_done_step=first_done_step, first_done_time_s=first_done_time_s)
    return EvalSeries(
        label=_safe_label(label or path.parent.name or path.stem),
        source=str(path),
        arrays=arrays,
        time_s=time_s,
        first_done_step=first_done_step,
        first_done_time_s=first_done_time_s,
        done_reason=np.zeros_like(first_done_step, dtype=np.int64),
        metadata={"input_kind": "timeseries_npz"},
    )


def _normalize_report_steps(first_done_step: np.ndarray, *, time_len: int) -> np.ndarray:
    """Convert older 1-based report step numbers to zero-based row indices."""
    steps = np.asarray(first_done_step, dtype=np.int64).copy()
    active = steps >= 0
    if time_len > 0 and np.any(active) and int(np.nanmax(steps[active])) >= time_len:
        steps[active] -= 1
    return steps


def _infer_time_s(
    arrays: dict[str, np.ndarray],
    *,
    first_done_step: np.ndarray,
    first_done_time_s: np.ndarray,
) -> np.ndarray:
    if "time_s" in arrays:
        return np.nanmean(_to_2d(arrays["time_s"], name="time_s"), axis=1)
    steps = next(iter(arrays.values())).shape[0] if arrays else 0
    valid = (first_done_step >= 0) & np.isfinite(first_done_time_s)
    if np.any(valid):
        dt_samples = first_done_time_s[valid] / (first_done_step[valid].astype(np.float64) + 1.0)
        dt_samples = dt_samples[np.isfinite(dt_samples) & (dt_samples > 0.0)]
        if dt_samples.size:
            dt = float(np.median(dt_samples))
            return (np.arange(steps, dtype=np.float64) + 1.0) * dt
    return np.arange(steps, dtype=np.float64)


def _load_recording_dir(path: Path, *, label: str | None = None) -> EvalSeries:
    from auvrl.scripts.demo._model_replay_io import DONE_REASON_NAMES, load_recording

    recording = load_recording(path)
    raw = recording.arrays
    episode_idx = np.asarray(raw["episode_idx"], dtype=np.int64)
    episodes = sorted(int(value) for value in np.unique(episode_idx))
    max_len = max(int(np.sum(episode_idx == episode)) for episode in episodes)
    scalar_names = [
        name
        for name, values in raw.items()
        if values.ndim == 1 and np.issubdtype(values.dtype, np.number)
    ]
    arrays = {
        name: np.full((max_len, len(episodes)), np.nan, dtype=np.float32)
        for name in scalar_names
        if name not in {"episode_idx"}
    }
    first_done_step = np.full(len(episodes), -1, dtype=np.int64)
    first_done_time_s = np.full(len(episodes), np.nan, dtype=np.float64)
    done_reason = np.zeros(len(episodes), dtype=np.int64)

    for column, episode in enumerate(episodes):
        indices = np.nonzero(episode_idx == episode)[0]
        for name in arrays:
            arrays[name][: len(indices), column] = np.asarray(raw[name][indices], dtype=np.float32)
        done_values = np.asarray(raw.get("would_done", raw.get("done_after_step"))[indices], dtype=bool)
        if np.any(done_values):
            local = int(np.argmax(done_values))
            first_done_step[column] = local
            first_done_time_s[column] = float(raw["time_s"][indices[local]])
            code = int(raw.get("would_done_reason_code", np.zeros_like(episode_idx))[indices[local]])
            done_reason[column] = code if code in DONE_REASON_NAMES else 0

    time_s = np.nanmax(arrays["time_s"], axis=1) if "time_s" in arrays else np.arange(max_len, dtype=np.float64)
    return EvalSeries(
        label=_safe_label(label or path.name),
        source=str(path),
        arrays=arrays,
        time_s=np.asarray(time_s, dtype=np.float64),
        first_done_step=first_done_step,
        first_done_time_s=first_done_time_s,
        done_reason=done_reason,
        metadata={"input_kind": "model_recording", **recording.manifest},
    )


def _default_time(arrays: dict[str, np.ndarray]) -> np.ndarray:
    if "time_s" in arrays:
        return np.nanmean(_to_2d(arrays["time_s"], name="time_s"), axis=1)
    steps = next(iter(arrays.values())).shape[0] if arrays else 0
    return np.arange(steps, dtype=np.float64)


def _step_to_time(first_done_step: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    result = np.full(first_done_step.shape, np.nan, dtype=np.float64)
    for index, step in enumerate(first_done_step):
        if step >= 0 and time_s.size:
            result[index] = float(time_s[min(int(step), time_s.size - 1)])
    return result


def _derive_first_done_step(arrays: dict[str, np.ndarray]) -> np.ndarray:
    candidate = None
    if "would_done" in arrays:
        candidate = arrays["would_done"] > 0.5
    elif "done_after_step" in arrays:
        candidate = arrays["done_after_step"] > 0.5
    elif "target_reached" in arrays:
        candidate = arrays["target_reached"] > 0.5
    if candidate is None:
        envs = next(iter(arrays.values())).shape[1] if arrays else 0
        return np.full(envs, -1, dtype=np.int64)
    first = np.full(candidate.shape[1], -1, dtype=np.int64)
    for env_idx in range(candidate.shape[1]):
        hits = np.nonzero(candidate[:, env_idx])[0]
        if hits.size:
            first[env_idx] = int(hits[0])
    return first


def evaluate_checkpoint(path: Path, args: argparse.Namespace, *, label: str) -> EvalSeries:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'torch'. Run through `uv run`.") from exc

    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends

    from auvrl.actuator.body_wrench_action import BodyWrenchAction
    from auvrl.scripts.demo import model_record
    from auvrl.tasks.roll import mdp
    from auvrl.tasks.roll.runtime import (
        current_root_pose_from_qpos,
        get_roll_task_state,
        quat_wxyz_to_roll_pitch_yaw,
    )

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    record_args = argparse.Namespace(
        policy="checkpoint",
        checkpoint=path,
        device=args.device,
        num_envs=args.num_envs,
        record_env_idx=0,
        episodes=1,
        max_steps=args.max_steps,
        post_done_steps=0,
        output_dir=None,
        overwrite=True,
        curriculum_stage=args.curriculum_stage,
        play_mode="training",
        deployment_completion_action="zero",
        episode_length_s=args.episode_length_s,
        roll_direction=args.roll_direction,
        print_period=0,
    )
    device = model_record._resolve_device(args.device)
    base_env = model_record._make_env(record_args, device)
    agent_cfg_dict = model_record._load_agent_cfg_dict(path)
    env = RslRlVecEnvWrapper(env=base_env, clip_actions=agent_cfg_dict.get("clip_actions"))
    try:
        policy = model_record._make_policy(record_args, env, path, agent_cfg_dict, device)
        env.reset()
        max_steps = args.max_steps
        if max_steps is None:
            max_steps = int(math.ceil(float(base_env.cfg.episode_length_s) / float(base_env.step_dt)))

        # Resolve env-dependent metadata once.
        wrench_term = base_env.action_manager.get_term("body_wrench")
        if not isinstance(wrench_term, BodyWrenchAction):
            raise RuntimeError("Expected body_wrench action term to be BodyWrenchAction.")
        try:
            hydro_term = base_env.action_manager.get_term("hydro")
        except KeyError:
            hydro_term = None
        thruster_names = tuple(getattr(wrench_term, "_site_names", ()) or ())
        if not thruster_names:
            thruster_names = tuple(
                f"thruster_{idx}" for idx in range(int(wrench_term.thruster_targets.shape[1]))
            )
        site_force_limit_n = float(
            getattr(wrench_term, "_site_force_limit_n", 0.0) or 0.0
        )
        reward_term_names = tuple(base_env.reward_manager.active_terms)
        try:
            roll_state_for_meta = get_roll_task_state(base_env)
            z_ref_initial = roll_state_for_meta.z_ref_m.detach().cpu().numpy().astype(np.float32)
        except Exception:  # pragma: no cover - defensive; metadata only.
            z_ref_initial = None

        action_dim_names = ("Fx_norm", "Fy_norm", "Fz_norm", "Mx_norm", "My_norm", "Mz_norm")

        series: dict[str, list[np.ndarray]] = {name: [] for name in TRAJECTORY_METRICS}
        series.update(
            {
                "reward": [],
                "phi_total_rad": [],
                "target_reached": [],
                "settle_counter_s": [],
                "x_drift_m": [],
                "y_drift_m": [],
                "z_drift_m": [],
                "xy_drift_m_peak": [],
                "would_done": [],
                "would_success": [],
                "roll_rad": [],
                "pitch_rad": [],
                "yaw_rad": [],
                "lin_vel_x_b": [],
                "lin_vel_y_b": [],
                "lin_vel_z_b": [],
                "ang_vel_x_b": [],
                "ang_vel_y_b": [],
                "ang_vel_z_b": [],
                "wrench_Fx": [],
                "wrench_Fy": [],
                "wrench_Fz": [],
                "wrench_Tx": [],
                "wrench_Ty": [],
                "wrench_Tz": [],
                "action_rate_l2": [],
                "thruster_max_abs_n": [],
                "hydro_wrench_norm": [],
            }
        )
        per_thruster_targets: list[np.ndarray] = []
        per_thruster_saturated: list[np.ndarray] = []
        per_reward_term: list[np.ndarray] = []
        per_action_dim: list[np.ndarray] = []

        first_done_step = np.full(base_env.num_envs, -1, dtype=np.int64)
        first_done_time_s = np.full(base_env.num_envs, np.nan, dtype=np.float64)
        done_reason = np.zeros(base_env.num_envs, dtype=np.int64)
        xy_peak = torch.zeros(base_env.num_envs, device=base_env.device)
        prev_action = torch.zeros(
            base_env.num_envs, int(wrench_term.action_dim), device=base_env.device
        )
        sat_eps = max(site_force_limit_n * 1.0e-6, 1.0e-6) if site_force_limit_n > 0 else 1.0e-6

        for step in range(max_steps):
            obs = env.get_observations()
            with torch.no_grad():
                actions = policy(obs)
            _obs_after, reward, done, _truncated, _extras = model_record.step_vec_env(env, actions)

            xy = mdp.xy_drift_m(base_env)
            xy_peak = torch.maximum(xy_peak, xy)

            # Signed RPY and z_drift from live qpos.
            robot = base_env.scene["robot"]
            root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
            roll_rad, pitch_rad, yaw_rad = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
            roll_state = get_roll_task_state(base_env)
            z_drift = root_pos_w[:, 2] - roll_state.z_ref_m

            lin_vel_b = mdp.base_lin_vel(base_env)
            ang_vel_b = mdp.base_ang_vel(base_env)

            desired_wrench = wrench_term.desired_wrench_b
            thr_targets = wrench_term.thruster_targets
            thr_max_abs = thr_targets.abs().amax(dim=1)
            if site_force_limit_n > 0:
                thr_saturated = (thr_targets.abs() >= (site_force_limit_n - sat_eps)).float()
            else:
                thr_saturated = torch.zeros_like(thr_targets)

            current_action = wrench_term.raw_action.detach()
            action_rate_sq = torch.sum((current_action - prev_action).square(), dim=1)
            prev_action = current_action.clone()

            values = {
                "reward": reward,
                "roll_progress_ratio": _metric_from_cfg(base_env, "roll_progress_ratio_last"),
                "phi_total_rad": mdp.phi_total_rad(base_env),
                "target_reached": mdp.target_reached(base_env),
                "settle_counter_s": mdp.settle_counter_s(base_env),
                "depth_abs_error_m": mdp.depth_abs_error_m(base_env),
                "xy_drift_m": xy,
                "x_drift_m": mdp.x_drift_m(base_env),
                "y_drift_m": mdp.y_drift_m(base_env),
                "z_drift_m": z_drift,
                "xy_drift_m_peak": xy_peak,
                "pitch_abs_rad": mdp.pitch_abs_rad(base_env),
                "yaw_abs_error_rad": mdp.yaw_abs_error_rad(base_env),
                "root_ang_speed_rad_s": mdp.root_ang_speed_rad_s(base_env),
                "body_wrench_action_l2": mdp.body_wrench_action_l2(base_env),
                "body_wrench_saturation_fraction": mdp.body_wrench_saturation_fraction(base_env),
                "would_done": done.float(),
                "would_success": _termination_value(base_env, "task_success").float(),
                "roll_rad": roll_rad,
                "pitch_rad": pitch_rad,
                "yaw_rad": yaw_rad,
                "lin_vel_x_b": lin_vel_b[:, 0],
                "lin_vel_y_b": lin_vel_b[:, 1],
                "lin_vel_z_b": lin_vel_b[:, 2],
                "ang_vel_x_b": ang_vel_b[:, 0],
                "ang_vel_y_b": ang_vel_b[:, 1],
                "ang_vel_z_b": ang_vel_b[:, 2],
                "wrench_Fx": desired_wrench[:, 0],
                "wrench_Fy": desired_wrench[:, 1],
                "wrench_Fz": desired_wrench[:, 2],
                "wrench_Tx": desired_wrench[:, 3],
                "wrench_Ty": desired_wrench[:, 4],
                "wrench_Tz": desired_wrench[:, 5],
                "action_rate_l2": action_rate_sq,
                "thruster_max_abs_n": thr_max_abs,
                "hydro_wrench_norm": mdp.hydro_wrench_norm(base_env) if hydro_term is not None else torch.zeros_like(reward),
            }
            for name, tensor in values.items():
                series.setdefault(name, []).append(tensor.detach().float().cpu().numpy())

            per_thruster_targets.append(thr_targets.detach().float().cpu().numpy())
            per_thruster_saturated.append(thr_saturated.detach().float().cpu().numpy())
            per_action_dim.append(current_action.detach().float().cpu().numpy())
            step_reward = getattr(base_env.reward_manager, "_step_reward", None)
            if step_reward is not None:
                per_reward_term.append(step_reward.detach().float().cpu().numpy())

            newly_done = (first_done_step < 0) & done.detach().cpu().numpy().astype(bool)
            if np.any(newly_done):
                first_done_step[newly_done] = step
                first_done_time_s[newly_done] = float((step + 1) * base_env.step_dt)
                reason_codes = _done_reason_codes(base_env).detach().cpu().numpy()
                done_reason[newly_done] = reason_codes[newly_done]
            if np.all(first_done_step >= 0):
                break

        arrays = {name: np.stack(values, axis=0).astype(np.float32) for name, values in series.items() if values}
        time_s = (np.arange(next(iter(arrays.values())).shape[0], dtype=np.float64) + 1.0) * float(base_env.step_dt)
        thruster_targets_arr = (
            np.stack(per_thruster_targets, axis=0).astype(np.float32) if per_thruster_targets else None
        )
        thruster_saturated_arr = (
            np.stack(per_thruster_saturated, axis=0).astype(np.float32) if per_thruster_saturated else None
        )
        reward_terms_arr = (
            np.stack(per_reward_term, axis=0).astype(np.float32) if per_reward_term else None
        )
        action_dims_arr = (
            np.stack(per_action_dim, axis=0).astype(np.float32) if per_action_dim else None
        )
        metadata = {
            "input_kind": "checkpoint",
            "checkpoint": str(path),
            "checkpoint_step": _checkpoint_step(path),
            "curriculum_stage": args.curriculum_stage,
            "num_envs": args.num_envs,
            "control_dt_s": float(base_env.step_dt),
            "site_force_limit_n": site_force_limit_n,
            "num_thrusters": int(thruster_targets_arr.shape[2]) if thruster_targets_arr is not None else 0,
        }
        if z_ref_initial is not None:
            metadata["z_ref_initial_m"] = z_ref_initial.tolist()
        return EvalSeries(
            label=label,
            source=str(path),
            arrays=arrays,
            time_s=time_s,
            first_done_step=first_done_step,
            first_done_time_s=first_done_time_s,
            done_reason=done_reason,
            metadata=metadata,
            thruster_targets_n=thruster_targets_arr,
            thruster_names=thruster_names,
            thruster_saturated=thruster_saturated_arr,
            reward_terms_step=reward_terms_arr,
            reward_term_names=reward_term_names,
            action_dims=action_dims_arr,
            action_dim_names=action_dim_names,
        )
    finally:
        env.close()


def _metric_from_cfg(env: Any, name: str) -> Any:
    cfg = env.cfg.metrics[name]
    return cfg.func(env, **getattr(cfg, "params", {}))


def _termination_value(env: Any, name: str) -> Any:
    try:
        return env.termination_manager.get_term(name)
    except KeyError:
        import torch

        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _done_reason_codes(env: Any) -> Any:
    import torch

    codes = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    for code, name in enumerate(OUTCOME_REASON_NAMES, start=1):
        if name == "multiple":
            continue
        active = _termination_value(env, name).bool()
        codes = torch.where(active & (codes == 0), torch.full_like(codes, code), codes)
        codes = torch.where(active & (codes != code), torch.full_like(codes, 6), codes)
    return codes


def write_artifacts(series: EvalSeries, output_dir: Path, *, skip_images: bool = False) -> dict[str, Any]:
    _ensure_output_dir(output_dir, overwrite=True)
    summary = summarize_eval(series)
    with (output_dir / "eval_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    with (output_dir / "eval_episodes.jsonl").open("w", encoding="utf-8") as file:
        for env_idx in range(series.num_envs):
            row = {
                "env_idx": env_idx,
                "first_done_step": int(series.first_done_step[env_idx]),
                "first_done_time_s": float(series.first_done_time_s[env_idx])
                if np.isfinite(series.first_done_time_s[env_idx])
                else None,
                "done_reason": int(series.done_reason[env_idx]) if series.done_reason.size else 0,
            }
            for metric in TERMINAL_METRICS:
                if metric in series.arrays:
                    row[f"final_{metric}"] = float(_last_active(series.arrays[metric], series.first_done_step)[env_idx])
            file.write(json.dumps(row, sort_keys=True) + "\n")
    npz_payload: dict[str, np.ndarray] = {
        "time_s": series.time_s,
        "first_done_step": series.first_done_step,
        "first_done_time_s": series.first_done_time_s,
        "done_reason": series.done_reason,
        **series.arrays,
    }
    if series.thruster_targets_n is not None:
        npz_payload["thruster_targets_n"] = series.thruster_targets_n
    if series.thruster_saturated is not None:
        npz_payload["thruster_saturated"] = series.thruster_saturated
    if series.reward_terms_step is not None:
        npz_payload["reward_terms_step"] = series.reward_terms_step
    if series.action_dims is not None:
        npz_payload["action_dims"] = series.action_dims
    if series.thruster_names:
        npz_payload["thruster_names"] = np.asarray(series.thruster_names)
    if series.reward_term_names:
        npz_payload["reward_term_names"] = np.asarray(series.reward_term_names)
    np.savez_compressed(output_dir / "eval_timeseries.npz", **npz_payload)
    _write_tensorboard(series, summary, output_dir)
    if not skip_images:
        image_path = output_dir / "summary_dashboard.png"
        _write_summary_image(series, summary, image_path)
        _add_image_to_tensorboard(image_path, output_dir, tag="EvalImage/summary_dashboard")
        _write_diagnostic_images(series, output_dir)
    return summary


def _write_tensorboard(series: EvalSeries, summary: dict[str, Any], output_dir: Path) -> None:
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError as exc:
        raise SystemExit("TensorBoard writer is not available. Run `uv sync`.") from exc

    step = int(series.metadata.get("checkpoint_step", 0) or 0)
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        writer.add_text("Eval/summary", _summary_markdown(summary), global_step=step)
        writer.add_scalar("EvalOutcome/done_rate", summary["done_rate"], step)
        for name, value in summary["outcome"].items():
            writer.add_scalar(f"EvalOutcome/{name}", value, step)
        if np.isfinite(summary["first_done_time_s"]["mean"]):
            writer.add_scalar("EvalTiming/first_done_time_s_mean", summary["first_done_time_s"]["mean"], step)
            writer.add_scalar("EvalTiming/first_done_time_s_p90", summary["first_done_time_s"]["p90"], step)
        for group, metrics in (("EvalTerminal", summary["terminal"]), ("EvalTrajectory", summary["trajectory"])):
            for name, stats in metrics.items():
                if isinstance(stats, dict):
                    for stat_name, value in stats.items():
                        if np.isfinite(value):
                            writer.add_scalar(f"{group}/{name}_{stat_name}", value, step)
        # Headline trajectory metrics (curated).
        for metric in TRAJECTORY_METRICS:
            if metric not in series.arrays:
                continue
            values = _mask_after_done(series.arrays[metric], series.first_done_step)
            for row_idx, time_value in enumerate(series.time_s[: values.shape[0]]):
                row = values[row_idx]
                writer.add_scalar(f"EvalTrajectoryMean/{metric}", float(np.nanmean(row)), row_idx)
                writer.add_scalar(f"EvalTrajectoryP90/{metric}", float(np.nanpercentile(row, 90)), row_idx)
                writer.add_scalar(f"EvalTrajectoryMax/{metric}", float(np.nanmax(row)), row_idx)
                writer.add_scalar("EvalTrajectoryTime/time_s", float(time_value), row_idx)
            writer.add_histogram(f"EvalHistogram/final_{metric}", _last_active(series.arrays[metric], series.first_done_step), step)
        # Extended trajectory metrics (signed RPY, drift components, body vels,
        # wrench components, action rate, etc.). Write mean ± p10/p90 over time.
        for metric in EXTENDED_TRAJECTORY_METRICS:
            if metric not in series.arrays:
                continue
            values = _mask_after_done(series.arrays[metric], series.first_done_step)
            for row_idx in range(values.shape[0]):
                row = values[row_idx]
                writer.add_scalar(f"EvalExtraMean/{metric}", float(np.nanmean(row)), row_idx)
                writer.add_scalar(f"EvalExtraP10/{metric}", float(np.nanpercentile(row, 10)), row_idx)
                writer.add_scalar(f"EvalExtraP90/{metric}", float(np.nanpercentile(row, 90)), row_idx)
            writer.add_histogram(f"EvalHistogram/final_{metric}", _last_active(series.arrays[metric], series.first_done_step), step)
        # Per-thruster commanded force time series and duty-cycle scalars.
        if series.thruster_targets_n is not None:
            thr = _mask_after_done_3d(series.thruster_targets_n, series.first_done_step)
            sat = (
                _mask_after_done_3d(series.thruster_saturated, series.first_done_step)
                if series.thruster_saturated is not None
                else None
            )
            num_thr = thr.shape[2]
            for i in range(num_thr):
                name = series.thruster_names[i] if i < len(series.thruster_names) else f"thruster_{i}"
                safe = _safe_label(name)
                channel = thr[:, :, i]
                for row_idx in range(channel.shape[0]):
                    writer.add_scalar(f"EvalThrusterMean/{safe}", float(np.nanmean(channel[row_idx])), row_idx)
                    writer.add_scalar(
                        f"EvalThrusterAbsMean/{safe}",
                        float(np.nanmean(np.abs(channel[row_idx]))),
                        row_idx,
                    )
                writer.add_histogram(
                    f"EvalThrusterHistogram/{safe}_final",
                    _last_active_3d(series.thruster_targets_n, series.first_done_step)[:, i],
                    step,
                )
                if sat is not None:
                    sat_channel = sat[:, :, i]
                    duty_cycle = float(np.nanmean(sat_channel))
                    writer.add_scalar(f"EvalThrusterDutyCycle/{safe}", duty_cycle, step)
        # Reward decomposition per term.
        if series.reward_terms_step is not None:
            rew = _mask_after_done_3d(series.reward_terms_step, series.first_done_step)
            for term_idx, term_name in enumerate(series.reward_term_names):
                safe = _safe_label(term_name)
                channel = rew[:, :, term_idx]
                for row_idx in range(channel.shape[0]):
                    writer.add_scalar(
                        f"EvalRewardTermMean/{safe}",
                        float(np.nanmean(channel[row_idx])),
                        row_idx,
                    )
                # Episode integral (sum over time, mean over envs).
                integral = float(np.nansum(np.nanmean(channel, axis=1)))
                writer.add_scalar(f"EvalRewardTermIntegral/{safe}", integral, step)
        # Per-action-dim histograms (final clipped action distribution per env).
        if series.action_dims is not None:
            final = _last_active_3d(series.action_dims, series.first_done_step)
            for dim_idx in range(final.shape[1]):
                dim_name = (
                    series.action_dim_names[dim_idx]
                    if dim_idx < len(series.action_dim_names)
                    else f"action_{dim_idx}"
                )
                writer.add_histogram(f"EvalActionHistogram/{_safe_label(dim_name)}", final[:, dim_idx], step)
    finally:
        writer.flush()
        writer.close()


def _mask_after_done_3d(values: np.ndarray | None, first_done_step: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros((0, 0, 0), dtype=np.float32)
    out = np.asarray(values, dtype=np.float64).copy()
    for env_idx, stop in enumerate(first_done_step):
        if stop >= 0:
            out[int(stop) + 1 :, env_idx, :] = np.nan
    return out


def _last_active_3d(values: np.ndarray | None, first_done_step: np.ndarray) -> np.ndarray:
    if values is None:
        return np.zeros((0, 0), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float64)
    n_envs = arr.shape[1]
    out = np.full((n_envs, arr.shape[2]), np.nan, dtype=np.float64)
    for env_idx, stop in enumerate(first_done_step):
        row = min(int(stop), arr.shape[0] - 1) if stop >= 0 else arr.shape[0] - 1
        out[env_idx, :] = arr[row, env_idx, :]
    return out


def _summary_markdown(summary: dict[str, Any]) -> str:
    timing = summary["first_done_time_s"]
    lines = [
        f"# {summary['label']}",
        "",
        f"- source: `{summary['source']}`",
        f"- envs: `{summary['num_envs']}`",
        f"- steps: `{summary['num_steps']}`",
        f"- done: `{summary['done_count']}/{summary['num_envs']}`",
        f"- mean first done: `{timing['mean']:.3f} s`" if np.isfinite(timing["mean"]) else "- mean first done: `n/a`",
        "",
        "## Outcome",
    ]
    for name, value in summary["outcome"].items():
        lines.append(f"- `{name}`: `{value:.3f}`")
    return "\n".join(lines)


def _add_image_to_tensorboard(
    image_path: Path,
    output_dir: Path,
    *,
    tag: str = "EvalImage/summary_dashboard",
    step: int = 0,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        return
    image = plt.imread(image_path)
    if image.ndim == 2:
        image = image[:, :, None]
    image_chw = np.moveaxis(image[:, :, :3], -1, 0)
    writer = SummaryWriter(log_dir=str(output_dir))
    try:
        writer.add_image(tag, image_chw, global_step=step)
    finally:
        writer.flush()
        writer.close()


def _write_diagnostic_images(series: EvalSeries, output_dir: Path) -> None:
    """Render extra diagnostic dashboards and add them to TensorBoard as images."""
    try:
        import matplotlib.pyplot as plt  # noqa: F401  (style sanity check only)
    except ModuleNotFoundError:
        return
    step = int(series.metadata.get("checkpoint_step", 0) or 0)
    plots = {
        "EvalImage/signed_rpy": (output_dir / "diagnostics_rpy_signed.png", _plot_signed_rpy_panel),
        "EvalImage/xy_trajectory": (output_dir / "diagnostics_xy_trajectory.png", _plot_xy_trajectory),
        "EvalImage/wrench_components": (output_dir / "diagnostics_wrench.png", _plot_wrench_components),
        "EvalImage/per_thruster": (output_dir / "diagnostics_thrusters.png", _plot_per_thruster_panel),
        "EvalImage/reward_decomposition": (
            output_dir / "diagnostics_reward_decomposition.png",
            _plot_reward_decomposition,
        ),
        "EvalImage/body_velocities": (output_dir / "diagnostics_body_vel.png", _plot_body_velocities),
        "EvalImage/action_histogram": (output_dir / "diagnostics_action_hist.png", _plot_action_histogram),
    }
    for tag, (path, plot_fn) in plots.items():
        try:
            ok = plot_fn(series, path)
        except Exception as exc:  # pragma: no cover - diagnostic safety net
            print(f"  [warn] diagnostic plot {tag} failed: {exc}")
            ok = False
        if ok and path.exists():
            _add_image_to_tensorboard(path, output_dir, tag=tag, step=step)


def _write_summary_image(series: EvalSeries, summary: dict[str, Any], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is needed for PNG summaries. Use --skip-images to skip.") from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(15, 9), dpi=150)
    grid = fig.add_gridspec(3, 3, height_ratios=[0.65, 1.0, 1.0])
    fig.patch.set_facecolor("#f8fafc")
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.78, series.label, fontsize=20, weight="bold", color="#172033")
    title_ax.text(0.0, 0.42, _headline(summary), fontsize=11, color="#46556b")
    title_ax.text(0.0, 0.12, str(series.source), fontsize=8, color="#64748b")

    _plot_metric(fig.add_subplot(grid[1, 0]), series, "roll_progress_ratio", "Roll progress", target=1.0)
    _plot_metric(fig.add_subplot(grid[1, 1]), series, "pitch_abs_rad", "Pitch abs rad")
    _plot_metric(fig.add_subplot(grid[1, 2]), series, "body_wrench_saturation_fraction", "Saturation", target=0.5)
    _plot_metric(fig.add_subplot(grid[2, 0]), series, "xy_drift_m", "XY drift m")
    _plot_metric(fig.add_subplot(grid[2, 1]), series, "yaw_abs_error_rad", "Yaw error rad")
    _plot_terminal_bars(fig.add_subplot(grid[2, 2]), summary)

    fig.tight_layout(pad=1.2)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _headline(summary: dict[str, Any]) -> str:
    done = f"done {summary['done_count']}/{summary['num_envs']}"
    timing = summary["first_done_time_s"]
    time_text = f"mean first done {timing['mean']:.2f}s" if np.isfinite(timing["mean"]) else "mean first done n/a"
    outcome = summary.get("outcome", {})
    success = outcome.get("success_rate", outcome.get("target_reached_rate"))
    success_text = f"success/reach {success:.0%}" if success is not None else "success/reach n/a"
    return f"{done} · {time_text} · {success_text}"


def _plot_metric(ax: Any, series: EvalSeries, metric: str, title: str, *, target: float | None = None) -> None:
    ax.set_title(title, loc="left", fontsize=11, weight="bold", color="#172033")
    if metric not in series.arrays:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#94a3b8")
        return
    values = _mask_after_done(series.arrays[metric], series.first_done_step)
    time_s = series.time_s[: values.shape[0]]
    mean = np.nanmean(values, axis=1)
    p10 = np.nanpercentile(values, 10, axis=1)
    p90 = np.nanpercentile(values, 90, axis=1)
    ax.fill_between(time_s, p10, p90, color="#93c5fd", alpha=0.22, linewidth=0)
    ax.plot(time_s, mean, color="#2563eb", linewidth=2.0)
    if target is not None:
        ax.axhline(target, color="#ef4444", linewidth=1.1, linestyle="--", alpha=0.75)
    ax.set_xlabel("time (s)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.25)


def _plot_terminal_bars(ax: Any, summary: dict[str, Any]) -> None:
    ax.set_title("Terminal snapshot", loc="left", fontsize=11, weight="bold", color="#172033")
    items = [
        ("pitch", "final_pitch_abs_rad"),
        ("yaw", "final_yaw_abs_error_rad"),
        ("xy", "final_xy_drift_m"),
        ("sat", "final_body_wrench_saturation_fraction"),
    ]
    labels: list[str] = []
    values: list[float] = []
    for label, key in items:
        stats = summary["terminal"].get(key)
        if stats and np.isfinite(stats["mean"]):
            labels.append(label)
            values.append(float(stats["mean"]))
    if not values:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", color="#94a3b8")
        return
    colors = ["#2563eb", "#0f766e", "#f97316", "#dc2626"][: len(values)]
    ax.bar(labels, values, color=colors, alpha=0.86)
    ax.tick_params(labelsize=8)
    ax.grid(True, axis="y", alpha=0.25)


# ---------------------------------------------------------------------------
# Diagnostic plots
# ---------------------------------------------------------------------------


def _episode_endpoint(series: EvalSeries, env_idx: int, arr2d: np.ndarray) -> int:
    stop = int(series.first_done_step[env_idx])
    if stop < 0:
        return arr2d.shape[0]
    return min(stop + 1, arr2d.shape[0])


def _plot_signed_rpy_panel(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if not all(name in series.arrays for name in SIGNED_RPY_METRICS):
        return False
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), dpi=140, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    colors = {"roll_rad": "#2563eb", "pitch_rad": "#0f766e", "yaw_rad": "#dc2626"}
    titles = {"roll_rad": "Roll φ (rad)", "pitch_rad": "Pitch θ (rad)", "yaw_rad": "Yaw ψ (rad)"}
    for ax, name in zip(axes, SIGNED_RPY_METRICS, strict=True):
        masked = _mask_after_done(series.arrays[name], series.first_done_step)
        time_s = series.time_s[: masked.shape[0]]
        mean = np.nanmean(masked, axis=1)
        p10 = np.nanpercentile(masked, 10, axis=1)
        p90 = np.nanpercentile(masked, 90, axis=1)
        ax.fill_between(time_s, p10, p90, color=colors[name], alpha=0.2, linewidth=0)
        ax.plot(time_s, mean, color=colors[name], linewidth=1.8)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle=":")
        ax.set_title(titles[name], loc="left", fontsize=11, weight="bold", color="#172033")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("time (s)", fontsize=9)
    fig.suptitle(f"{series.label} — Signed roll/pitch/yaw", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_xy_trajectory(series: EvalSeries, path: Path, *, max_paths: int = 64) -> bool:
    import matplotlib.pyplot as plt

    if "x_drift_m" not in series.arrays or "y_drift_m" not in series.arrays:
        return False
    x = np.asarray(series.arrays["x_drift_m"], dtype=np.float64)
    y = np.asarray(series.arrays["y_drift_m"], dtype=np.float64)
    num_envs = x.shape[1]
    paths_to_draw = min(num_envs, max_paths)
    fig, (ax_path, ax_density) = plt.subplots(1, 2, figsize=(13, 6), dpi=140)
    fig.patch.set_facecolor("#f8fafc")

    cmap = plt.cm.viridis(np.linspace(0.0, 1.0, paths_to_draw))
    for env_idx in range(paths_to_draw):
        end = _episode_endpoint(series, env_idx, x)
        xs = x[:end, env_idx]
        ys = y[:end, env_idx]
        ax_path.plot(xs, ys, color=cmap[env_idx], linewidth=0.9, alpha=0.65)
        ax_path.scatter(xs[-1:], ys[-1:], color=cmap[env_idx], s=14, edgecolor="white", linewidths=0.5)
    ax_path.scatter([0], [0], color="#dc2626", marker="x", s=80, linewidths=2.2, label="reset reference")
    ax_path.set_title(
        f"XY drift paths (first {paths_to_draw}/{num_envs} envs)",
        loc="left",
        fontsize=11,
        weight="bold",
        color="#172033",
    )
    ax_path.set_xlabel("x drift (m)", fontsize=9)
    ax_path.set_ylabel("y drift (m)", fontsize=9)
    ax_path.set_aspect("equal", adjustable="datalim")
    ax_path.grid(True, alpha=0.25)
    ax_path.tick_params(labelsize=8)
    ax_path.legend(loc="lower right", fontsize=8, frameon=False)

    # Final-position scatter density over all envs.
    final_x = _last_active(series.arrays["x_drift_m"], series.first_done_step)
    final_y = _last_active(series.arrays["y_drift_m"], series.first_done_step)
    finite = np.isfinite(final_x) & np.isfinite(final_y)
    if np.any(finite):
        ax_density.scatter(
            final_x[finite],
            final_y[finite],
            c="#2563eb",
            alpha=0.55,
            s=22,
            edgecolor="white",
            linewidths=0.6,
        )
        ax_density.axhline(0.0, color="#94a3b8", linewidth=0.8, linestyle=":")
        ax_density.axvline(0.0, color="#94a3b8", linewidth=0.8, linestyle=":")
    ax_density.set_title("Final XY position per env", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_density.set_xlabel("x drift (m)", fontsize=9)
    ax_density.set_ylabel("y drift (m)", fontsize=9)
    ax_density.set_aspect("equal", adjustable="datalim")
    ax_density.grid(True, alpha=0.25)
    ax_density.tick_params(labelsize=8)

    fig.suptitle(f"{series.label} — XY drift", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_wrench_components(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    components = WRENCH_COMPONENT_METRICS
    if not any(name in series.arrays for name in components):
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=140, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    color_force = "#2563eb"
    color_torque = "#dc2626"
    for ax, name in zip(axes.flat, components, strict=True):
        if name not in series.arrays:
            ax.axis("off")
            continue
        masked = _mask_after_done(series.arrays[name], series.first_done_step)
        time_s = series.time_s[: masked.shape[0]]
        mean = np.nanmean(masked, axis=1)
        p10 = np.nanpercentile(masked, 10, axis=1)
        p90 = np.nanpercentile(masked, 90, axis=1)
        is_force = name.startswith("wrench_F")
        color = color_force if is_force else color_torque
        ax.fill_between(time_s, p10, p90, color=color, alpha=0.18, linewidth=0)
        ax.plot(time_s, mean, color=color, linewidth=1.6)
        ax.axhline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
        unit = "N" if is_force else "N·m"
        ax.set_title(f"{name} [{unit}]", loc="left", fontsize=10, weight="bold", color="#172033")
        ax.grid(True, alpha=0.25)
        ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=9)
    fig.suptitle(f"{series.label} — Body-frame wrench command", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_per_thruster_panel(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if series.thruster_targets_n is None:
        return False
    targets = _mask_after_done_3d(series.thruster_targets_n, series.first_done_step)
    n_thr = targets.shape[2]
    time_s = series.time_s[: targets.shape[0]]

    fig = plt.figure(figsize=(14, 9), dpi=140)
    fig.patch.set_facecolor("#f8fafc")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.4, 1.0])

    # Top-left: per-thruster mean command time series.
    ax_ts = fig.add_subplot(grid[0, 0])
    cmap = plt.cm.tab10(np.linspace(0, 1, max(n_thr, 1)))
    for i in range(n_thr):
        name = series.thruster_names[i] if i < len(series.thruster_names) else f"t{i}"
        ax_ts.plot(time_s, np.nanmean(targets[:, :, i], axis=1), color=cmap[i % len(cmap)], linewidth=1.2, label=name)
    ax_ts.axhline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
    limit = float(series.metadata.get("site_force_limit_n", 0.0) or 0.0)
    if limit > 0:
        ax_ts.axhline(limit, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.6, label=f"±{limit:g} N")
        ax_ts.axhline(-limit, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.6)
    ax_ts.set_title("Per-thruster mean command (N)", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_ts.set_xlabel("time (s)", fontsize=9)
    ax_ts.grid(True, alpha=0.25)
    ax_ts.tick_params(labelsize=8)
    ax_ts.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    # Top-right: heatmap of |force| (time x thruster), mean across envs.
    ax_hm = fig.add_subplot(grid[0, 1])
    heat = np.nanmean(np.abs(targets), axis=1).T  # (n_thr, T)
    im = ax_hm.imshow(
        heat,
        aspect="auto",
        origin="lower",
        extent=(float(time_s[0]) if time_s.size else 0.0, float(time_s[-1]) if time_s.size else 0.0, -0.5, n_thr - 0.5),
        cmap="magma",
    )
    ax_hm.set_yticks(np.arange(n_thr))
    yticklabels = [
        series.thruster_names[i] if i < len(series.thruster_names) else f"t{i}"
        for i in range(n_thr)
    ]
    ax_hm.set_yticklabels(yticklabels, fontsize=7)
    ax_hm.set_title("Per-thruster |F| heatmap (mean across envs)", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_hm.set_xlabel("time (s)", fontsize=9)
    plt.colorbar(im, ax=ax_hm, fraction=0.04, pad=0.02, label="|F| (N)")

    # Bottom-left: duty-cycle bar (fraction of episode at the limit).
    ax_duty = fig.add_subplot(grid[1, 0])
    if series.thruster_saturated is not None:
        sat = _mask_after_done_3d(series.thruster_saturated, series.first_done_step)
        duty = np.nanmean(sat.reshape(-1, n_thr), axis=0)
    else:
        duty = np.zeros(n_thr)
    ax_duty.bar(np.arange(n_thr), duty, color="#dc2626", alpha=0.78)
    ax_duty.set_xticks(np.arange(n_thr))
    ax_duty.set_xticklabels(yticklabels, fontsize=7, rotation=30, ha="right")
    ax_duty.set_ylim(0.0, max(0.1, float(np.nanmax(duty) * 1.15)))
    ax_duty.set_title("Per-thruster saturation duty cycle", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_duty.set_ylabel("fraction of steps at limit", fontsize=8)
    ax_duty.grid(True, axis="y", alpha=0.25)

    # Bottom-right: peak |F| per thruster.
    ax_peak = fig.add_subplot(grid[1, 1])
    peak = np.nanmax(np.abs(targets).reshape(-1, n_thr), axis=0)
    ax_peak.bar(np.arange(n_thr), peak, color="#f97316", alpha=0.85)
    ax_peak.set_xticks(np.arange(n_thr))
    ax_peak.set_xticklabels(yticklabels, fontsize=7, rotation=30, ha="right")
    if limit > 0:
        ax_peak.axhline(limit, color="#dc2626", linewidth=1.0, linestyle="--", alpha=0.7, label=f"limit {limit:g} N")
        ax_peak.legend(loc="upper right", fontsize=7, frameon=False)
    ax_peak.set_title("Per-thruster peak |F|", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_peak.set_ylabel("peak |F| (N)", fontsize=8)
    ax_peak.grid(True, axis="y", alpha=0.25)

    fig.suptitle(f"{series.label} — Per-thruster diagnostics", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_reward_decomposition(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if series.reward_terms_step is None or not series.reward_term_names:
        return False
    rewards = _mask_after_done_3d(series.reward_terms_step, series.first_done_step)
    time_s = series.time_s[: rewards.shape[0]]
    n_terms = rewards.shape[2]
    term_mean_t = np.nanmean(rewards, axis=1)  # (T, n_terms)

    fig, (ax_lines, ax_bars) = plt.subplots(1, 2, figsize=(14, 6), dpi=140, gridspec_kw={"width_ratios": [1.6, 1.0]})
    fig.patch.set_facecolor("#f8fafc")

    cmap = plt.cm.tab20(np.linspace(0, 1, max(n_terms, 1)))
    for i, name in enumerate(series.reward_term_names):
        ax_lines.plot(time_s, term_mean_t[:, i], color=cmap[i % len(cmap)], linewidth=1.2, label=name)
    ax_lines.axhline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
    ax_lines.set_title("Per-term reward rate (mean across envs)", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_lines.set_xlabel("time (s)", fontsize=9)
    ax_lines.grid(True, alpha=0.25)
    ax_lines.tick_params(labelsize=8)
    ax_lines.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    integrals = np.nansum(term_mean_t, axis=0) * float(series.metadata.get("control_dt_s", 1.0) or 1.0)
    order = np.argsort(integrals)
    ax_bars.barh(np.arange(n_terms), integrals[order], color=cmap[order % len(cmap)], alpha=0.85)
    ax_bars.set_yticks(np.arange(n_terms))
    ax_bars.set_yticklabels([series.reward_term_names[i] for i in order], fontsize=8)
    ax_bars.axvline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
    ax_bars.set_title("Episode-mean reward integral per term", loc="left", fontsize=11, weight="bold", color="#172033")
    ax_bars.set_xlabel("∫ r·dt (mean env, mean episode)", fontsize=9)
    ax_bars.grid(True, axis="x", alpha=0.25)

    fig.suptitle(f"{series.label} — Reward decomposition", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_body_velocities(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    have_lin = all(name in series.arrays for name in LIN_VEL_METRICS)
    have_ang = all(name in series.arrays for name in ANG_VEL_METRICS)
    if not (have_lin or have_ang):
        return False
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), dpi=140, sharex=True)
    fig.patch.set_facecolor("#f8fafc")
    rows = [
        ("linear body vel (m/s)", LIN_VEL_METRICS, "#2563eb"),
        ("angular body vel (rad/s)", ANG_VEL_METRICS, "#0f766e"),
    ]
    for row_idx, (title, names, color) in enumerate(rows):
        for col_idx, name in enumerate(names):
            ax = axes[row_idx, col_idx]
            if name not in series.arrays:
                ax.axis("off")
                continue
            masked = _mask_after_done(series.arrays[name], series.first_done_step)
            time_s = series.time_s[: masked.shape[0]]
            mean = np.nanmean(masked, axis=1)
            p10 = np.nanpercentile(masked, 10, axis=1)
            p90 = np.nanpercentile(masked, 90, axis=1)
            ax.fill_between(time_s, p10, p90, color=color, alpha=0.18, linewidth=0)
            ax.plot(time_s, mean, color=color, linewidth=1.6)
            ax.axhline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
            short = name.replace("_b", "")
            ax.set_title(f"{title}: {short}", loc="left", fontsize=10, weight="bold", color="#172033")
            ax.grid(True, alpha=0.25)
            ax.tick_params(labelsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)", fontsize=9)
    fig.suptitle(f"{series.label} — Body velocity components", fontsize=13, weight="bold", color="#172033")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_action_histogram(series: EvalSeries, path: Path) -> bool:
    import matplotlib.pyplot as plt

    if series.action_dims is None:
        return False
    final = _last_active_3d(series.action_dims, series.first_done_step)
    n_dim = final.shape[1]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6), dpi=140)
    fig.patch.set_facecolor("#f8fafc")
    for idx, ax in enumerate(axes.flat):
        if idx >= n_dim:
            ax.axis("off")
            continue
        values = final[:, idx]
        values = values[np.isfinite(values)]
        if values.size == 0:
            ax.axis("off")
            continue
        ax.hist(values, bins=24, color="#2563eb", alpha=0.85)
        ax.axvline(0.0, color="#94a3b8", linewidth=0.7, linestyle=":")
        ax.axvline(1.0, color="#dc2626", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.axvline(-1.0, color="#dc2626", linewidth=0.7, linestyle="--", alpha=0.6)
        name = (
            series.action_dim_names[idx]
            if idx < len(series.action_dim_names)
            else f"a{idx}"
        )
        ax.set_title(name, loc="left", fontsize=10, weight="bold", color="#172033")
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.2)
    fig.suptitle(
        f"{series.label} — Final clipped policy action distribution",
        fontsize=12,
        weight="bold",
        color="#172033",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    return True


def run(args: argparse.Namespace) -> list[tuple[Path, dict[str, Any]]]:
    suite = _make_suite_name(args.suite)
    output_root = args.output_root.expanduser().resolve() / suite
    checkpoints = _resolve_checkpoints(args)
    input_count = len(checkpoints) + len(args.recording_dir) + len(args.timeseries_npz)
    if input_count == 0:
        raise SystemExit("Provide at least one --checkpoint, --checkpoint-glob, --recording-dir, or --timeseries-npz.")
    if args.label is not None and input_count != 1:
        raise SystemExit("--label is only allowed when there is exactly one input.")
    _ensure_output_dir(output_root, overwrite=args.overwrite)

    results: list[tuple[Path, dict[str, Any]]] = []
    for checkpoint in checkpoints:
        label = _safe_label(args.label or _checkpoint_label(checkpoint))
        series = evaluate_checkpoint(checkpoint, args, label=label)
        run_dir = output_root / label
        _ensure_output_dir(run_dir, overwrite=args.overwrite)
        results.append((run_dir, write_artifacts(series, run_dir, skip_images=args.skip_images)))
    for path in args.recording_dir:
        label = _safe_label(args.label or path.name)
        series = _load_recording_dir(path.expanduser(), label=label)
        run_dir = output_root / label
        _ensure_output_dir(run_dir, overwrite=args.overwrite)
        results.append((run_dir, write_artifacts(series, run_dir, skip_images=args.skip_images)))
    for path in args.timeseries_npz:
        label = _safe_label(args.label or path.parent.name or path.stem)
        series = _load_timeseries_npz(path.expanduser(), label=label)
        run_dir = output_root / label
        _ensure_output_dir(run_dir, overwrite=args.overwrite)
        results.append((run_dir, write_artifacts(series, run_dir, skip_images=args.skip_images)))
    _write_suite_index(output_root, results)
    return results


def _write_suite_index(output_root: Path, results: Iterable[tuple[Path, dict[str, Any]]]) -> None:
    rows = []
    for run_dir, summary in results:
        rows.append(
            {
                "label": summary["label"],
                "run_dir": str(run_dir),
                "source": summary["source"],
                "done_rate": summary["done_rate"],
                "mean_first_done_time_s": summary["first_done_time_s"]["mean"],
                "outcome": summary["outcome"],
            }
        )
    with (output_root / "suite_index.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tensorboard_command": f"uv run tensorboard --logdir {output_root}",
                "runs": rows,
            },
            file,
            indent=2,
            sort_keys=True,
        )
        file.write("\n")


def main() -> None:
    args = _parse_args()
    results = run(args)
    suite_dir = results[0][0].parent if results else args.output_root
    print(f"TensorBoard eval suite written: {suite_dir}")
    print(f"  tensorboard: uv run tensorboard --logdir {suite_dir}")
    for run_dir, summary in results:
        timing = summary["first_done_time_s"]
        print(
            f"  {summary['label']}: done={summary['done_count']}/{summary['num_envs']} "
            f"mean_first_done={timing['mean']:.3f}s run_dir={run_dir}"
        )


if __name__ == "__main__":
    main()
