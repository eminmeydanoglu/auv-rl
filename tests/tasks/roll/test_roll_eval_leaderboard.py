from __future__ import annotations

import json
from pathlib import Path

from auvrl.scripts.eval.roll_eval_leaderboard import collect_rows, render_markdown, write_leaderboard


def _write_summary(path: Path, *, label: str, success_rate: float, xy_peak: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "label": label,
                "source": f"{label}.pt",
                "num_envs": 8,
                "done_rate": 1.0,
                "first_done_time_s": {"mean": 9.0},
                "metadata": {"curriculum_stage": "c3_test", "checkpoint_step": 99},
                "outcome": {"success_rate": success_rate, "task_success_rate": success_rate},
                "terminal": {
                    "final_xy_drift_m_peak": {"mean": xy_peak},
                    "final_pitch_abs_rad": {"mean": 0.2},
                    "final_yaw_abs_error_rad": {"mean": 0.3},
                    "final_depth_abs_error_m": {"mean": 0.1},
                },
                "trajectory": {
                    "xy_drift_m": {"time_mean_of_env_mean": 0.4},
                    "body_wrench_saturation_fraction": {"time_mean_of_env_mean": 0.5},
                },
            }
        ),
        encoding="utf-8",
    )


def test_collect_rows_sorts_by_success_then_xy(tmp_path: Path) -> None:
    _write_summary(tmp_path / "suite" / "slow" / "eval_summary.json", label="slow", success_rate=0.5, xy_peak=0.1)
    _write_summary(tmp_path / "suite" / "best" / "eval_summary.json", label="best", success_rate=1.0, xy_peak=0.2)
    _write_summary(tmp_path / "suite" / "tie" / "eval_summary.json", label="tie", success_rate=1.0, xy_peak=0.1)

    rows = collect_rows(tmp_path)

    assert [row.label for row in rows] == ["tie", "best", "slow"]
    assert rows[0].suite == "suite"
    assert rows[0].stage == "c3_test"


def test_write_leaderboard_creates_csv_and_markdown(tmp_path: Path) -> None:
    _write_summary(tmp_path / "suite" / "run" / "eval_summary.json", label="run", success_rate=1.0, xy_peak=0.2)

    rows = write_leaderboard(
        search_root=tmp_path,
        csv_path=tmp_path / "leaderboard.csv",
        markdown_path=tmp_path / "leaderboard.md",
    )
    markdown = render_markdown(rows)

    assert len(rows) == 1
    assert (tmp_path / "leaderboard.csv").exists()
    assert (tmp_path / "leaderboard.md").exists()
    assert "success_rate" in markdown
    assert "run" in markdown
