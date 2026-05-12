"""Run a roll checkpoint and plot per-timestep aggregate telemetry."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime
import math
import os
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'matplotlib'. Install project deps first or run with "
        "`uv run --with matplotlib ...`."
    ) from exc

try:
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit("Missing dependency 'numpy'. Install project deps first.") from exc

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.managers.metrics_manager import MetricsTermCfg  # type: ignore[import-not-found]
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab RL dependencies. Ensure mjlab is available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    ROLL_CURRICULUM_STAGES,
    get_roll_curriculum_stage,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.tasks.roll.runtime import (  # noqa: E402  # type: ignore[import-not-found]
    current_root_ang_vel_b_from_qvel,
    current_root_pose_from_qpos,
    get_roll_task_state,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CHECKPOINT = (
    ROOT
    / "logs/rsl_rl/taluy_roll_v1/"
    / "2026-05-05_12-26-21_c3e_720_hold_0p05_xy_light_from_b_from_a650_env1024_steps256_hold05_20260505_122601"
    / "model_699.pt"
)
DEFAULT_OUTPUT_ROOT = ROOT / "reports" / "roll_timeseries"
SUMMARY_SUFFIXES = ("mean", "std", "p10", "p50", "p90", "min", "max")


def _roll_rate_abs_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 0].abs()


def _pitch_rate_abs_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 1].abs()


def _yaw_rate_abs_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 2].abs()


def _roll_rate_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 0]


def _pitch_rate_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 1]


def _yaw_rate_rad_s(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    return current_root_ang_vel_b_from_qvel(robot)[:, 2]


def _x_drift_m(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    state = get_roll_task_state(env)
    return root_pos_w[:, 0] - state.xy_ref_w[:, 0]


def _y_drift_m(env: ManagerBasedRlEnv) -> torch.Tensor:
    robot = env.scene["robot"]
    root_pos_w, _quat_wxyz = current_root_pose_from_qpos(robot)
    state = get_roll_task_state(env)
    return root_pos_w[:, 1] - state.xy_ref_w[:, 1]


METRIC_ALIASES = {
    "roll_progress_ratio_last": "roll_progress_ratio",
    "phi_total_rad_last": "phi_total_rad",
    "target_reached_last": "target_reached",
    "settle_counter_s_last": "settle_counter_s",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to the trained roll PPO checkpoint.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default="c3e_720_hold_0p05_xy_light",
        help="Roll curriculum stage used for the evaluation environment.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device. 'auto' picks cuda if available, else cpu.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=256,
        help="Number of deterministic parallel environments.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Rollout length in control steps. Defaults to the stage episode length.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Override stage episode length.",
    )
    parser.add_argument(
        "--roll-direction",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Roll direction for the evaluation environment.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to reports/roll_timeseries/<stamp>_c3e_model699.",
    )
    parser.add_argument(
        "--stop-when-all-done",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop the rollout once every original env has terminated or timed out.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _load_agent_cfg_dict(checkpoint_path: Path) -> dict[str, Any]:
    params_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not params_path.exists():
        return asdict(taluy_roll_ppo_runner_cfg())

    with params_path.open("r", encoding="utf-8") as file:
        cfg = yaml.full_load(file)
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"Expected mapping in {params_path}, got {type(cfg).__name__}."
        )
    return cfg


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    checkpoint_stem = args.checkpoint_file.stem
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return DEFAULT_OUTPUT_ROOT / f"{stamp}_{args.curriculum_stage}_{checkpoint_stem}"


def _as_numpy(values: torch.Tensor) -> np.ndarray:
    return values.detach().float().cpu().numpy()


def _summarize(values: np.ndarray, active: np.ndarray) -> dict[str, float]:
    active_values = values[active]
    if active_values.size == 0:
        return {suffix: math.nan for suffix in SUMMARY_SUFFIXES}
    return {
        "mean": float(np.mean(active_values)),
        "std": float(np.std(active_values)),
        "p10": float(np.percentile(active_values, 10)),
        "p50": float(np.percentile(active_values, 50)),
        "p90": float(np.percentile(active_values, 90)),
        "min": float(np.min(active_values)),
        "max": float(np.max(active_values)),
    }


def _scalar(value: np.ndarray, active: np.ndarray, suffix: str = "mean") -> float:
    return _summarize(value, active)[suffix]


def _make_env(
    args: argparse.Namespace,
    device: str,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
    cfg = make_taluy_roll_env_cfg(
        num_envs=args.num_envs,
        curriculum_stage=args.curriculum_stage,
        episode_length_s=args.episode_length_s,
        roll_direction=args.roll_direction,
    )
    cfg.metrics.update(
        {
            "roll_rate_abs_rad_s": MetricsTermCfg(func=_roll_rate_abs_rad_s),
            "pitch_rate_abs_rad_s": MetricsTermCfg(func=_pitch_rate_abs_rad_s),
            "yaw_rate_abs_rad_s": MetricsTermCfg(func=_yaw_rate_abs_rad_s),
            "roll_rate_rad_s": MetricsTermCfg(func=_roll_rate_rad_s),
            "pitch_rate_rad_s": MetricsTermCfg(func=_pitch_rate_rad_s),
            "yaw_rate_rad_s": MetricsTermCfg(func=_yaw_rate_rad_s),
            "x_drift_m": MetricsTermCfg(func=_x_drift_m),
            "y_drift_m": MetricsTermCfg(func=_y_drift_m),
        }
    )
    base_env = ManagerBasedRlEnv(cfg=cfg, device=device)
    agent_cfg_dict = _load_agent_cfg_dict(args.checkpoint_file)
    env = RslRlVecEnvWrapper(
        env=base_env,
        clip_actions=agent_cfg_dict.get("clip_actions", 1.0),
    )
    return base_env, env


def _collect_metrics(
    base_env: ManagerBasedRlEnv,
    reward: torch.Tensor,
) -> dict[str, np.ndarray]:
    # MetricsManager computes before auto-reset, so its step values preserve the
    # terminal frame. Direct qpos/qvel reads after env.step() would see reset state.
    step_values = getattr(base_env.metrics_manager, "_step_values")
    result = {
        "reward": _as_numpy(reward),
    }
    for idx, name in enumerate(base_env.metrics_manager.active_terms):
        key = METRIC_ALIASES.get(name, name)
        result[key] = _as_numpy(step_values[:, idx])
    return result


def _write_summary_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_final_env_csv(
    path: Path,
    first_done_step: np.ndarray,
    first_done_time: np.ndarray,
    final_metrics: dict[str, np.ndarray],
) -> None:
    fieldnames = ["env_idx", "first_done_step", "first_done_time_s", *final_metrics]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for env_idx in range(first_done_step.shape[0]):
            row: dict[str, float | int] = {
                "env_idx": env_idx,
                "first_done_step": int(first_done_step[env_idx]),
                "first_done_time_s": float(first_done_time[env_idx]),
            }
            for name, values in final_metrics.items():
                row[name] = float(values[env_idx])
            writer.writerow(row)


def _plot_series(
    rows: list[dict[str, float]],
    output_dir: Path,
    *,
    target_roll_rad: float,
) -> list[Path]:
    t = np.array([row["time_s"] for row in rows], dtype=float)

    def series(name: str, suffix: str = "mean") -> np.ndarray:
        return np.array([row.get(f"{name}_{suffix}", math.nan) for row in rows])

    def first_time_for(
        name: str,
        threshold: float,
        suffix: str | None = "mean",
    ) -> float | None:
        if suffix is None:
            values = np.array([row.get(name, math.nan) for row in rows])
        else:
            values = series(name, suffix)
        hits = np.nonzero(values >= threshold)[0]
        if hits.size == 0:
            return None
        return float(t[int(hits[0])])

    target_time_s = first_time_for("target_reached", 0.5)
    done_time_s = first_time_for("first_done_fraction", 1.0, suffix=None)

    def plot_mean_band(ax: Any, name: str, label: str, color: str) -> None:
        mean = series(name, "mean")
        p10 = series(name, "p10")
        p90 = series(name, "p90")
        ax.plot(t, mean, label=label, color=color, linewidth=2.0)
        ax.fill_between(t, p10, p90, color=color, alpha=0.16, linewidth=0)

    def annotate_roll_events(ax: Any) -> None:
        if target_time_s is not None:
            ax.axvline(
                target_time_s,
                color="#15803d",
                linestyle=":",
                linewidth=1.5,
                label="target_reached flips",
            )
        if done_time_s is not None:
            ax.axvline(
                done_time_s,
                color="#7c2d12",
                linestyle="-.",
                linewidth=1.3,
                label="all envs done",
            )

    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "roll_progress_ratio", "progress ratio", "#2563eb")
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.1, label="target")
    annotate_roll_events(ax)
    ax.scatter(t[-1], series("roll_progress_ratio", "mean")[-1], color="#2563eb", zorder=4)
    ax.annotate(
        f"final {series('roll_progress_ratio', 'mean')[-1]:.3f}",
        xy=(t[-1], series("roll_progress_ratio", "mean")[-1]),
        xytext=(-84, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#2563eb"},
        fontsize=9,
    )
    ax.set_title("Roll Progress Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signed progress / target")
    ax.set_xlim(max(0.0, float(t[0]) - 0.05), float(t[-1]) + 0.12)
    ax.set_ylim(-0.04, max(1.08, float(np.nanmax(series("roll_progress_ratio", "mean"))) + 0.04))
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "01_roll_progress_ratio.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "roll_progress_ratio", "progress ratio", "#2563eb")
    ax.axhline(1.0, color="#111827", linestyle="--", linewidth=1.1, label="target")
    annotate_roll_events(ax)
    ax.set_title("Roll Progress Near Target")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("signed progress / target")
    ax.set_xlim(max(0.0, float(t[-1]) - 0.55), float(t[-1]) + 0.04)
    ax.set_ylim(0.90, max(1.025, float(np.nanmax(series("roll_progress_ratio", "mean"))) + 0.006))
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "01b_roll_progress_ratio_zoom.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "phi_total_rad", "phi_total", "#0891b2")
    ax.axhline(target_roll_rad, color="#111827", linestyle="--", linewidth=1.1, label="720 target")
    annotate_roll_events(ax)
    ax.set_title("Cumulative Roll Angle Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("phi_total [rad]")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "02_phi_total_rad.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "root_ang_speed_rad_s", "root angular speed", "#dc2626")
    plot_mean_band(ax, "roll_rate_abs_rad_s", "|roll rate|", "#f97316")
    plot_mean_band(ax, "pitch_rate_abs_rad_s", "|pitch rate|", "#7c3aed")
    plot_mean_band(ax, "yaw_rate_abs_rad_s", "|yaw rate|", "#059669")
    ax.set_title("Angular Speeds Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad/s")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "03_angular_speeds.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "xy_drift_m", "XY drift", "#16a34a")
    plot_mean_band(ax, "depth_abs_error_m", "depth abs error", "#0f766e")
    ax.set_title("Position Drift Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("m")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "04_position_drift.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "pitch_abs_rad", "|pitch|", "#9333ea")
    plot_mean_band(ax, "yaw_abs_error_rad", "|yaw error|", "#db2777")
    ax.set_title("Attitude Errors Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("rad")
    ax.grid(True, alpha=0.25)
    ax.legend()
    path = output_dir / "05_attitude_errors.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "settle_counter_s", "settle counter", "#1d4ed8")
    ax2 = ax.twinx()
    ax2.plot(t, series("target_reached", "mean"), color="#15803d", linewidth=1.8, label="target reached fraction")
    ax.set_title("Target Reached And Settle Counter")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("settle counter [s]")
    ax2.set_ylabel("fraction")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    path = output_dir / "06_target_and_settle.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    plot_mean_band(ax, "body_wrench_saturation_fraction", "thruster saturation fraction", "#ca8a04")
    ax2 = ax.twinx()
    ax2.plot(t, series("body_wrench_action_l2", "mean"), color="#475569", linewidth=1.7, label="action L2")
    ax.set_title("Action Saturation Over Time")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("fraction")
    ax2.set_ylabel("normalized action L2")
    ax.grid(True, alpha=0.25)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    path = output_dir / "07_action_saturation.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def _write_report(
    path: Path,
    *,
    args: argparse.Namespace,
    output_dir: Path,
    rows: list[dict[str, float]],
    first_done_step: np.ndarray,
    first_done_time: np.ndarray,
    final_metrics: dict[str, np.ndarray],
    plot_paths: list[Path],
    target_roll_rad: float,
) -> None:
    final_row = rows[-1]
    done_mask = first_done_step >= 0
    done_count = int(done_mask.sum())
    done_times = first_done_time[done_mask]
    median_done_s = float(np.median(done_times)) if done_times.size else math.nan
    p90_done_s = float(np.percentile(done_times, 90)) if done_times.size else math.nan

    def final(name: str, suffix: str = "mean") -> float:
        return float(final_row.get(f"{name}_{suffix}", math.nan))

    lines = [
        "# Taluy Roll C3E Time-Series Evaluation",
        "",
        f"- checkpoint: `{args.checkpoint_file}`",
        f"- stage: `{args.curriculum_stage}`",
        f"- num_envs: `{args.num_envs}`",
        f"- simulated control steps: `{len(rows)}`",
        f"- control_dt_s: `{rows[0]['time_s']:.6f}`",
        f"- target_roll_rad: `{target_roll_rad:.6f}`",
        f"- done envs: `{done_count}/{args.num_envs}`",
        f"- median first done time: `{_fmt(median_done_s)} s`",
        f"- p90 first done time: `{_fmt(p90_done_s)} s`",
        "",
        "## How To Read `root_ang_speed_rad_s`",
        "",
        "`root_ang_speed_rad_s` is the magnitude of body-frame angular velocity: "
        "`sqrt(wx^2 + wy^2 + wz^2)`. In TensorBoard it is logged as an episode "
        "metric aggregate, not as one single instant. In this report the time-series "
        "plots show the per-timestep env mean with p10-p90 bands.",
        "",
        "## Final Active-Step Summary",
        "",
        "| metric | mean | p10 | p90 |",
        "|---|---:|---:|---:|",
    ]
    for name in (
        "roll_progress_ratio",
        "phi_total_rad",
        "root_ang_speed_rad_s",
        "roll_rate_abs_rad_s",
        "xy_drift_m",
        "depth_abs_error_m",
        "pitch_abs_rad",
        "yaw_abs_error_rad",
        "settle_counter_s",
        "body_wrench_saturation_fraction",
    ):
        lines.append(
            f"| `{name}` | {_fmt(final(name))} | {_fmt(final(name, 'p10'))} | {_fmt(final(name, 'p90'))} |"
        )

    if done_count:
        lines.extend(
            [
                "",
                "## Metrics At First Done",
                "",
                "| metric | mean | p10 | p90 |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, values in final_metrics.items():
            selected = values[done_mask]
            lines.append(
                f"| `{name}` | {_fmt(float(np.mean(selected)))} | "
                f"{_fmt(float(np.percentile(selected, 10)))} | "
                f"{_fmt(float(np.percentile(selected, 90)))} |"
            )

    lines.extend(
        [
            "",
            "## Mean Across The Rollout",
            "",
            "| metric | time-mean of env mean | peak env mean | final env mean |",
            "|---|---:|---:|---:|",
        ]
    )
    for name in (
        "root_ang_speed_rad_s",
        "roll_rate_abs_rad_s",
        "xy_drift_m",
        "depth_abs_error_m",
        "pitch_abs_rad",
        "yaw_abs_error_rad",
        "body_wrench_saturation_fraction",
    ):
        values = np.array([row[f"{name}_mean"] for row in rows], dtype=float)
        lines.append(
            f"| `{name}` | {_fmt(float(np.mean(values)))} | "
            f"{_fmt(float(np.max(values)))} | {_fmt(float(values[-1]))} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- summary CSV: `{output_dir / 'timeseries_summary.csv'}`",
            f"- per-env first-done CSV: `{output_dir / 'per_env_first_done.csv'}`",
            f"- raw arrays: `{output_dir / 'timeseries_raw.npz'}`",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"- plot: `{plot_path}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.checkpoint_file = args.checkpoint_file.resolve()
    if not args.checkpoint_file.exists():
        raise SystemExit(f"Checkpoint file not found: {args.checkpoint_file}")
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.steps is not None and args.steps <= 0:
        raise SystemExit("--steps must be positive.")

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()
    device = _resolve_device(args.device)
    output_dir = _output_dir(args).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stage = get_roll_curriculum_stage(args.curriculum_stage)
    target_roll_rad = math.radians(stage.target_roll_deg)
    base_env, env = _make_env(args, device)
    agent_cfg_dict = _load_agent_cfg_dict(args.checkpoint_file)
    steps = args.steps
    if steps is None:
        horizon_s = args.episode_length_s or stage.episode_length_s
        steps = max(1, math.ceil(horizon_s / float(base_env.step_dt)))

    rows: list[dict[str, float]] = []
    raw: dict[str, list[np.ndarray]] = {}
    first_done_step = np.full(args.num_envs, -1, dtype=np.int64)
    first_done_time = np.full(args.num_envs, np.nan, dtype=np.float64)
    final_metrics: dict[str, np.ndarray] = {}
    alive = np.ones(args.num_envs, dtype=bool)

    try:
        runner = MjlabOnPolicyRunner(env, agent_cfg_dict, device=device)
        runner.load(
            str(args.checkpoint_file),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        env.reset()
        obs = env.get_observations()
        reward = torch.zeros(args.num_envs, device=base_env.device)

        for step in range(1, steps + 1):
            active_before = alive.copy()
            with torch.no_grad():
                action = policy(obs)
            step_result = env.step(action)
            if len(step_result) == 4:
                obs, reward, done, _info = step_result
            else:
                obs, reward, terminated, truncated, _info = step_result
                done = terminated | truncated

            metrics = _collect_metrics(base_env, reward)
            done_np = np.asarray(done.detach().cpu().numpy(), dtype=bool)
            first_done_now = active_before & done_np
            if first_done_now.any():
                first_done_step[first_done_now] = step
                first_done_time[first_done_now] = step * float(base_env.step_dt)
                for name, values in metrics.items():
                    if name not in final_metrics:
                        final_metrics[name] = np.full(
                            args.num_envs, np.nan, dtype=np.float64
                        )
                    final_metrics[name][first_done_now] = values[first_done_now]

            row: dict[str, float] = {
                "step": float(step),
                "time_s": step * float(base_env.step_dt),
                "alive_count": float(active_before.sum()),
                "first_done_count": float((first_done_step >= 0).sum()),
                "first_done_fraction": float((first_done_step >= 0).mean()),
            }
            for name, values in metrics.items():
                raw.setdefault(name, []).append(values.copy())
                for suffix, value in _summarize(values, active_before).items():
                    row[f"{name}_{suffix}"] = value
            rows.append(row)

            alive &= ~done_np
            if args.stop_when_all_done and not alive.any():
                break

        not_done = first_done_step < 0
        if not_done.any():
            for name, values in _collect_metrics(base_env, reward).items():
                if name not in final_metrics:
                    final_metrics[name] = np.full(
                        args.num_envs, np.nan, dtype=np.float64
                    )
                final_metrics[name][not_done] = values[not_done]

        _write_summary_csv(output_dir / "timeseries_summary.csv", rows)
        _write_final_env_csv(
            output_dir / "per_env_first_done.csv",
            first_done_step,
            first_done_time,
            final_metrics,
        )
        np.savez_compressed(
            output_dir / "timeseries_raw.npz",
            **{name: np.stack(values) for name, values in raw.items()},
            first_done_step=first_done_step,
            first_done_time_s=first_done_time,
        )
        plot_paths = _plot_series(rows, output_dir, target_roll_rad=target_roll_rad)
        _write_report(
            output_dir / "report.md",
            args=args,
            output_dir=output_dir,
            rows=rows,
            first_done_step=first_done_step,
            first_done_time=first_done_time,
            final_metrics=final_metrics,
            plot_paths=plot_paths,
            target_roll_rad=target_roll_rad,
        )

        final_row = rows[-1]
        print("Taluy roll time-series evaluation complete.")
        print(f"  device={device}")
        print(f"  checkpoint={args.checkpoint_file}")
        print(f"  stage={args.curriculum_stage}")
        print(f"  num_envs={args.num_envs}")
        print(f"  steps={len(rows)}")
        print(f"  output_dir={output_dir}")
        print(
            "  final mean: "
            f"progress={final_row['roll_progress_ratio_mean']:.3f}, "
            f"phi={final_row['phi_total_rad_mean']:.3f} rad, "
            f"ang_speed={final_row['root_ang_speed_rad_s_mean']:.3f} rad/s, "
            f"xy={final_row['xy_drift_m_mean']:.3f} m, "
            f"settle={final_row['settle_counter_s_mean']:.3f} s"
        )
        print(
            "  first_done: "
            f"{int((first_done_step >= 0).sum())}/{args.num_envs}"
        )
    finally:
        base_env.close()


if __name__ == "__main__":
    main()
