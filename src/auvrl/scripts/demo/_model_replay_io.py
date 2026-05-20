"""Shared recording I/O helpers for model replay tools."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

FORMAT_VERSION = "auvrl_model_replay_v1"
LEGACY_FORMATS = {"taluy_roll_replay_v1"}

TRAJECTORY_FILE = "trajectory.npz"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "episode_summary.jsonl"

DONE_REASON_CODES = {
    "none": 0,
    "task_success": 1,
    "excess_pitch": 2,
    "excess_depth_error": 3,
    "excess_xy_drift": 4,
    "time_out": 5,
    "multiple": 9,
}
DONE_REASON_NAMES = {value: key for key, value in DONE_REASON_CODES.items()}


@dataclass(frozen=True)
class ModelRecording:
    """Loaded model replay recording."""

    root: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    episode_summaries: list[dict[str, Any]]

    @property
    def num_frames(self) -> int:
        if "time_s" not in self.arrays:
            return 0
        return int(self.arrays["time_s"].shape[0])

    @property
    def control_dt_s(self) -> float:
        return float(self.manifest.get("control_dt_s", 0.008))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def write_recording(
    output_dir: Path,
    *,
    manifest: dict[str, Any],
    arrays: dict[str, np.ndarray],
    episode_summaries: list[dict[str, Any]],
    overwrite: bool = False,
) -> None:
    """Write a complete recording directory."""
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory already exists and is not empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = validate_arrays(arrays)
    manifest = dict(manifest)
    manifest.setdefault("format", FORMAT_VERSION)
    manifest["recorded_frames"] = frame_count

    with (output_dir / MANIFEST_FILE).open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True, default=_json_default)
        file.write("\n")

    np.savez_compressed(output_dir / TRAJECTORY_FILE, **arrays)  # pyright: ignore[reportArgumentType]

    with (output_dir / SUMMARY_FILE).open("w", encoding="utf-8") as file:
        for summary in episode_summaries:
            file.write(json.dumps(summary, default=_json_default, sort_keys=True) + "\n")


def load_recording(recording_dir: Path) -> ModelRecording:
    """Load a recording directory produced by model_record.py."""
    recording_dir = recording_dir.resolve()
    manifest_path = recording_dir / MANIFEST_FILE
    trajectory_path = recording_dir / TRAJECTORY_FILE
    summary_path = recording_dir / SUMMARY_FILE

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {MANIFEST_FILE}: {manifest_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing {TRAJECTORY_FILE}: {trajectory_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if not isinstance(manifest, dict):
        raise ValueError(f"Expected manifest mapping in {manifest_path}.")

    fmt = manifest.get("format")
    if fmt != FORMAT_VERSION and fmt not in LEGACY_FORMATS:
        raise ValueError(
            f"Unsupported recording format {fmt!r}. Expected {FORMAT_VERSION!r}."
        )

    data = np.load(trajectory_path, allow_pickle=False)
    arrays = {name: data[name] for name in data.files}
    validate_arrays(arrays)

    episode_summaries: list[dict[str, Any]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    episode_summaries.append(json.loads(line))

    return ModelRecording(
        root=recording_dir,
        manifest=manifest,
        arrays=arrays,
        episode_summaries=episode_summaries,
    )


def validate_arrays(arrays: dict[str, np.ndarray]) -> int:
    """Validate that all trajectory arrays share the same frame dimension."""
    if "time_s" not in arrays:
        raise ValueError("Recording arrays must include 'time_s'.")
    frame_count = int(arrays["time_s"].shape[0])
    for name, values in arrays.items():
        if values.shape[0] != frame_count:
            raise ValueError(
                f"Array {name!r} has {values.shape[0]} frames; expected {frame_count}."
            )
    return frame_count


def done_reason_code(reasons: list[str]) -> int:
    """Encode zero, one, or multiple done reasons as an integer code."""
    if not reasons:
        return DONE_REASON_CODES["none"]
    if len(reasons) > 1:
        return DONE_REASON_CODES["multiple"]
    return DONE_REASON_CODES.get(reasons[0], DONE_REASON_CODES["multiple"])


def done_reason_name(code: int) -> str:
    """Return a stable display name for a done reason code."""
    return DONE_REASON_NAMES.get(int(code), f"unknown:{int(code)}")
