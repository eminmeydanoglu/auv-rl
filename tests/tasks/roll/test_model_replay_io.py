from __future__ import annotations

import json

import numpy as np

from auvrl.scripts.demo._model_replay_io import (
    FORMAT_VERSION,
    done_reason_code,
    load_recording,
    write_recording,
)


def test_model_replay_io_roundtrip(tmp_path) -> None:
    arrays = {
        "time_s": np.asarray([0.0, 0.008], dtype=np.float64),
        "root_pos_w": np.zeros((2, 3), dtype=np.float32),
        "quat_wxyz": np.tile(
            np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            (2, 1),
        ),
    }
    manifest = {
        "format": FORMAT_VERSION,
        "control_dt_s": 0.008,
        "reward_term_names": ["roll_progress"],
    }
    summaries = [{"episode_idx": 0, "frames": 2}]

    write_recording(
        tmp_path,
        manifest=manifest,
        arrays=arrays,
        episode_summaries=summaries,
    )

    loaded = load_recording(tmp_path)
    assert loaded.num_frames == 2
    assert loaded.control_dt_s == 0.008
    assert loaded.episode_summaries == summaries
    assert np.array_equal(loaded.arrays["root_pos_w"], arrays["root_pos_w"])

    with (tmp_path / "manifest.json").open("r", encoding="utf-8") as file:
        written_manifest = json.load(file)
    assert written_manifest["format"] == FORMAT_VERSION
    assert written_manifest["recorded_frames"] == 2


def test_done_reason_code_handles_none_single_and_multiple() -> None:
    assert done_reason_code([]) == 0
    assert done_reason_code(["task_success"]) == 1
    assert done_reason_code(["task_success", "time_out"]) == 9

