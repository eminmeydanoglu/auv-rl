from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from auvrl.scripts.eval.taluy_roll_tensorboard import (
    EvalSeries,
    _infer_time_s,
    _last_active_3d,
    _load_timeseries_npz,
    _mask_after_done_3d,
    _normalize_report_steps,
    _plot_action_histogram,
    _plot_per_thruster_panel,
    _plot_reward_decomposition,
    _plot_signed_rpy_panel,
    _plot_xy_trajectory,
    summarize_eval,
    write_artifacts,
)


def test_summarize_eval_uses_first_done_and_final_values() -> None:
    arrays = {
        "roll_progress_ratio": np.asarray([[0.1, 0.2], [1.0, 0.9], [1.1, 1.2]], dtype=np.float32),
        "pitch_abs_rad": np.asarray([[0.01, 0.02], [0.03, 0.04], [0.05, 0.06]], dtype=np.float32),
        "body_wrench_saturation_fraction": np.asarray(
            [[0.0, 0.1], [0.5, 0.25], [0.75, 0.5]],
            dtype=np.float32,
        ),
        "target_reached": np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32),
    }
    series = EvalSeries(
        label="unit",
        source="test",
        arrays=arrays,
        time_s=np.asarray([0.1, 0.2, 0.3]),
        first_done_step=np.asarray([1, -1]),
        first_done_time_s=np.asarray([0.2, np.nan]),
        done_reason=np.asarray([1, 0]),
        metadata={},
    )

    summary = summarize_eval(series)

    assert summary["done_count"] == 1
    assert summary["done_rate"] == 0.5
    assert summary["terminal"]["first_done_pitch_abs_rad"]["mean"] == np.float32(0.01)
    assert np.isclose(summary["terminal"]["final_roll_progress_ratio"]["mean"], 0.65)
    assert summary["trajectory"]["body_wrench_saturation_fraction"]["peak_env_mean"] == 0.5


def test_load_timeseries_npz_preserves_first_done(tmp_path: Path) -> None:
    path = tmp_path / "timeseries_raw.npz"
    np.savez_compressed(
        path,
        roll_progress_ratio=np.asarray([[0.0, 0.0], [1.0, 0.8]], dtype=np.float32),
        pitch_abs_rad=np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
        first_done_step=np.asarray([1, -1], dtype=np.int64),
        first_done_time_s=np.asarray([0.2, np.nan], dtype=np.float64),
    )

    series = _load_timeseries_npz(path, label="loaded")

    assert series.label == "loaded"
    assert series.num_steps == 2
    assert series.num_envs == 2
    assert series.first_done_step.tolist() == [1, -1]


def test_normalize_report_steps_converts_one_based_final_step() -> None:
    steps = _normalize_report_steps(np.asarray([1144, 1143, -1]), time_len=1144)

    assert steps.tolist() == [1143, 1142, -1]


def test_infer_time_s_from_first_done_time() -> None:
    time_s = _infer_time_s(
        {"pitch_abs_rad": np.zeros((4, 2), dtype=np.float32)},
        first_done_step=np.asarray([3, -1]),
        first_done_time_s=np.asarray([0.032, np.nan]),
    )

    assert np.allclose(time_s, [0.008, 0.016, 0.024, 0.032])


def test_write_artifacts_creates_json_npz_and_image(tmp_path: Path) -> None:
    arrays = {
        "roll_progress_ratio": np.asarray([[0.1, 0.2], [1.0, 0.9]], dtype=np.float32),
        "pitch_abs_rad": np.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=np.float32),
        "body_wrench_saturation_fraction": np.asarray([[0.0, 0.1], [0.5, 0.25]], dtype=np.float32),
        "xy_drift_m": np.asarray([[0.0, 0.0], [0.3, 0.4]], dtype=np.float32),
    }
    series = EvalSeries(
        label="artifact",
        source="test",
        arrays=arrays,
        time_s=np.asarray([0.1, 0.2]),
        first_done_step=np.asarray([1, 1]),
        first_done_time_s=np.asarray([0.2, 0.2]),
        done_reason=np.asarray([1, 1]),
        metadata={"eval_kind": "current"},
    )

    write_artifacts(series, tmp_path, skip_images=False)

    assert (tmp_path / "eval_summary.json").exists()
    assert (tmp_path / "eval_episodes.jsonl").exists()
    assert (tmp_path / "eval_timeseries.npz").exists()
    assert (tmp_path / "summary_dashboard.png").exists()
    summary = json.loads((tmp_path / "eval_summary.json").read_text())
    assert summary["metadata"]["eval_kind"] == "current"


def test_mask_after_done_3d_zeros_after_first_done() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    masked = _mask_after_done_3d(arr, np.asarray([1, -1]))

    assert np.allclose(masked[:1, 0, :], arr[:1, 0, :])
    assert np.all(np.isnan(masked[1:, 0, :]))
    assert np.allclose(masked[:, 1, :], arr[:, 1, :])


def test_last_active_3d_picks_done_row_for_each_env() -> None:
    arr = np.arange(24, dtype=np.float32).reshape(4, 2, 3)
    last = _last_active_3d(arr, np.asarray([0, 3]))

    assert np.allclose(last[0], arr[0, 0])
    assert np.allclose(last[1], arr[2, 1])


def _make_rich_series(num_steps: int = 6, num_envs: int = 4, num_thr: int = 8) -> EvalSeries:
    rng = np.random.default_rng(0)
    arrays = {
        "x_drift_m": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "y_drift_m": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "roll_rad": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "pitch_rad": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "yaw_rad": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "lin_vel_x_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "lin_vel_y_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "lin_vel_z_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "ang_vel_x_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "ang_vel_y_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "ang_vel_z_b": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Fx": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Fy": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Fz": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Tx": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Ty": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "wrench_Tz": rng.normal(size=(num_steps, num_envs)).astype(np.float32),
        "action_rate_l2": rng.uniform(size=(num_steps, num_envs)).astype(np.float32),
    }
    thruster_targets = rng.uniform(-40.0, 40.0, size=(num_steps, num_envs, num_thr)).astype(np.float32)
    thruster_saturated = (np.abs(thruster_targets) > 35.0).astype(np.float32)
    reward_terms = rng.normal(size=(num_steps, num_envs, 3)).astype(np.float32)
    action_dims = rng.uniform(-1.0, 1.0, size=(num_steps, num_envs, 6)).astype(np.float32)
    return EvalSeries(
        label="rich",
        source="test",
        arrays=arrays,
        time_s=(np.arange(num_steps) + 1) * 0.05,
        first_done_step=np.asarray([num_steps - 1, num_steps - 2, -1, 1]),
        first_done_time_s=np.asarray([num_steps * 0.05, (num_steps - 1) * 0.05, np.nan, 0.1]),
        done_reason=np.asarray([1, 2, 0, 3]),
        metadata={
            "control_dt_s": 0.05,
            "site_force_limit_n": 40.0,
            "num_thrusters": num_thr,
        },
        thruster_targets_n=thruster_targets,
        thruster_names=tuple(f"t{i}" for i in range(num_thr)),
        thruster_saturated=thruster_saturated,
        reward_terms_step=reward_terms,
        reward_term_names=("roll_progress", "xy_drift", "thruster_saturation"),
        action_dims=action_dims,
        action_dim_names=("Fx_norm", "Fy_norm", "Fz_norm", "Mx_norm", "My_norm", "Mz_norm"),
    )


def test_diagnostic_plots_emit_pngs(tmp_path: Path) -> None:
    series = _make_rich_series()
    assert _plot_signed_rpy_panel(series, tmp_path / "rpy.png")
    assert _plot_xy_trajectory(series, tmp_path / "xy.png")
    assert _plot_per_thruster_panel(series, tmp_path / "thrusters.png")
    assert _plot_reward_decomposition(series, tmp_path / "rewards.png")
    assert _plot_action_histogram(series, tmp_path / "actions.png")
    for name in ("rpy.png", "xy.png", "thrusters.png", "rewards.png", "actions.png"):
        assert (tmp_path / name).exists()


def test_write_artifacts_persists_per_thruster_and_reward_arrays(tmp_path: Path) -> None:
    series = _make_rich_series()
    write_artifacts(series, tmp_path, skip_images=False)
    with np.load(tmp_path / "eval_timeseries.npz") as data:
        assert "thruster_targets_n" in data
        assert data["thruster_targets_n"].shape == series.thruster_targets_n.shape
        assert "reward_terms_step" in data
        assert data["reward_terms_step"].shape == series.reward_terms_step.shape
        assert "action_dims" in data
        assert list(data["thruster_names"]) == list(series.thruster_names)
        assert list(data["reward_term_names"]) == list(series.reward_term_names)
    # Diagnostic PNGs should have been generated alongside the summary dashboard.
    for name in (
        "summary_dashboard.png",
        "diagnostics_rpy_signed.png",
        "diagnostics_xy_trajectory.png",
        "diagnostics_thrusters.png",
        "diagnostics_reward_decomposition.png",
        "diagnostics_body_vel.png",
        "diagnostics_wrench.png",
        "diagnostics_action_hist.png",
    ):
        assert (tmp_path / name).exists(), name
