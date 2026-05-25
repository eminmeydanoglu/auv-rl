from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SEARCH_ROOT = Path(__file__).resolve().parents[4] / "logs" / "tensorboard_eval" / "roll"


COLUMNS = (
    "rank",
    "suite",
    "label",
    "stage",
    "checkpoint_step",
    "success_rate",
    "task_success_rate",
    "done_rate",
    "first_done_time_s_mean",
    "xy_peak_mean_m",
    "xy_time_mean_m",
    "pitch_peak_mean_rad",
    "yaw_peak_mean_rad",
    "depth_peak_mean_m",
    "saturation_time_mean",
    "num_envs",
    "summary_path",
    "source",
)


@dataclass(frozen=True)
class LeaderboardRow:
    suite: str
    label: str
    stage: str
    checkpoint_step: int | None
    success_rate: float | None
    task_success_rate: float | None
    done_rate: float | None
    first_done_time_s_mean: float | None
    xy_peak_mean_m: float | None
    xy_time_mean_m: float | None
    pitch_peak_mean_rad: float | None
    yaw_peak_mean_rad: float | None
    depth_peak_mean_m: float | None
    saturation_time_mean: float | None
    num_envs: int | None
    summary_path: Path
    source: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a roll eval leaderboard from eval_summary.json files.")
    parser.add_argument("--search-root", type=Path, default=DEFAULT_SEARCH_ROOT)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--append-markdown", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--suite", default=None)
    return parser.parse_args()


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        out = float(value)
        return out if math.isfinite(out) else None
    return None


def _nested_float(data: dict[str, Any], *path: str) -> float | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return _finite_float(value)


def _nested_int(data: dict[str, Any], *path: str) -> int | None:
    value: Any = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return int(value)
    return None


def _suite_name(path: Path, search_root: Path) -> str:
    try:
        rel = path.parent.relative_to(search_root)
    except ValueError:
        rel = path.parent
    parts = rel.parts
    if len(parts) >= 2:
        return parts[0]
    if len(parts) == 1:
        return search_root.name
    return path.parent.parent.name


def row_from_summary(path: Path, search_root: Path) -> LeaderboardRow:
    with path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    metadata = summary.get("metadata", {}) if isinstance(summary.get("metadata"), dict) else {}
    return LeaderboardRow(
        suite=_suite_name(path, search_root),
        label=str(summary.get("label") or path.parent.name),
        stage=str(metadata.get("curriculum_stage") or ""),
        checkpoint_step=_nested_int(summary, "metadata", "checkpoint_step"),
        success_rate=_nested_float(summary, "outcome", "success_rate"),
        task_success_rate=_nested_float(summary, "outcome", "task_success_rate"),
        done_rate=_nested_float(summary, "done_rate"),
        first_done_time_s_mean=_nested_float(summary, "first_done_time_s", "mean"),
        xy_peak_mean_m=_nested_float(summary, "terminal", "final_xy_drift_m_peak", "mean"),
        xy_time_mean_m=_nested_float(summary, "trajectory", "xy_drift_m", "time_mean_of_env_mean"),
        pitch_peak_mean_rad=_nested_float(summary, "terminal", "final_pitch_abs_rad", "mean"),
        yaw_peak_mean_rad=_nested_float(summary, "terminal", "final_yaw_abs_error_rad", "mean"),
        depth_peak_mean_m=_nested_float(summary, "terminal", "final_depth_abs_error_m", "mean"),
        saturation_time_mean=_nested_float(
            summary,
            "trajectory",
            "body_wrench_saturation_fraction",
            "time_mean_of_env_mean",
        ),
        num_envs=_nested_int(summary, "num_envs"),
        summary_path=path,
        source=str(summary.get("source") or ""),
    )


def collect_rows(search_root: Path, *, suite: str | None = None) -> list[LeaderboardRow]:
    root = search_root.expanduser().resolve()
    paths = sorted(root.glob("**/eval_summary.json"))
    rows = [row_from_summary(path, root) for path in paths]
    if suite is not None:
        rows = [row for row in rows if row.suite == suite]
    return sort_rows(rows)


def _sort_value(value: float | int | None, *, reverse: bool) -> float:
    if value is None:
        return -math.inf if reverse else math.inf
    return float(value)


def sort_rows(rows: list[LeaderboardRow]) -> list[LeaderboardRow]:
    return sorted(
        rows,
        key=lambda row: (
            -_sort_value(row.success_rate, reverse=True),
            -_sort_value(row.task_success_rate, reverse=True),
            -_sort_value(row.done_rate, reverse=True),
            _sort_value(row.xy_peak_mean_m, reverse=False),
            _sort_value(row.saturation_time_mean, reverse=False),
            _sort_value(row.first_done_time_s_mean, reverse=False),
            row.suite,
            row.label,
        ),
    )


def _format_value(value: float | int | str | Path | None) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.6g}"
    return value


def _row_dict(rank: int, row: LeaderboardRow) -> dict[str, str]:
    values = {
        "rank": rank,
        "suite": row.suite,
        "label": row.label,
        "stage": row.stage,
        "checkpoint_step": row.checkpoint_step,
        "success_rate": row.success_rate,
        "task_success_rate": row.task_success_rate,
        "done_rate": row.done_rate,
        "first_done_time_s_mean": row.first_done_time_s_mean,
        "xy_peak_mean_m": row.xy_peak_mean_m,
        "xy_time_mean_m": row.xy_time_mean_m,
        "pitch_peak_mean_rad": row.pitch_peak_mean_rad,
        "yaw_peak_mean_rad": row.yaw_peak_mean_rad,
        "depth_peak_mean_m": row.depth_peak_mean_m,
        "saturation_time_mean": row.saturation_time_mean,
        "num_envs": row.num_envs,
        "summary_path": row.summary_path,
        "source": row.source,
    }
    return {key: _format_value(values[key]) for key in COLUMNS}


def write_csv(rows: list[LeaderboardRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(_row_dict(rank, row))


def render_markdown(rows: list[LeaderboardRow], *, title: str = "Roll eval leaderboard") -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    display_columns = (
        "rank",
        "suite",
        "label",
        "stage",
        "checkpoint_step",
        "success_rate",
        "task_success_rate",
        "done_rate",
        "first_done_time_s_mean",
        "xy_peak_mean_m",
        "saturation_time_mean",
    )
    lines = [f"# {title}", "", f"Generated: `{created_at}`", "", f"Runs: `{len(rows)}`", ""]
    lines.append("| " + " | ".join(display_columns) + " |")
    lines.append("| " + " | ".join("---" for _ in display_columns) + " |")
    for rank, row in enumerate(rows, start=1):
        data = _row_dict(rank, row)
        lines.append("| " + " | ".join(data[column] for column in display_columns) + " |")
    lines.append("")
    return "\n".join(lines)


def write_markdown(rows: list[LeaderboardRow], path: Path, *, title: str = "Roll eval leaderboard") -> str:
    text = render_markdown(rows, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def append_markdown(rows: list[LeaderboardRow], path: Path, *, title: str = "Roll eval leaderboard") -> None:
    text = render_markdown(rows, title=title)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write("\n\n")
        file.write(text)


def write_leaderboard(
    *,
    search_root: Path,
    csv_path: Path,
    markdown_path: Path,
    append_markdown_path: Path | None = None,
    limit: int | None = None,
    suite: str | None = None,
) -> list[LeaderboardRow]:
    rows = collect_rows(search_root, suite=suite)
    if limit is not None:
        rows = rows[:limit]
    write_csv(rows, csv_path)
    write_markdown(rows, markdown_path)
    if append_markdown_path is not None:
        append_markdown(rows, append_markdown_path)
    return rows


def main() -> None:
    args = _parse_args()
    root = args.search_root.expanduser().resolve()
    csv_path = args.csv.expanduser() if args.csv is not None else root / "leaderboard.csv"
    markdown_path = args.markdown.expanduser() if args.markdown is not None else root / "leaderboard.md"
    rows = write_leaderboard(
        search_root=root,
        csv_path=csv_path,
        markdown_path=markdown_path,
        append_markdown_path=args.append_markdown.expanduser() if args.append_markdown is not None else None,
        limit=args.limit,
        suite=args.suite,
    )
    print(f"Wrote {len(rows)} rows")
    print(f"  csv: {csv_path}")
    print(f"  markdown: {markdown_path}")


if __name__ == "__main__":
    main()
