from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from auvrl.scripts.train import taluy_roll as roll_train


def _eval_args() -> SimpleNamespace:
    return SimpleNamespace(
        eval_interval=1,
        eval_continue_on_error=False,
        eval_curriculum_stage=None,
        auto_curriculum=None,
        auto_curriculum_goal_stage="c3q_720_c3l_strict_settle",
        curriculum_stage="c3l_720_xy_guard",
    )


def _best_safe_args(**overrides: object) -> SimpleNamespace:
    args = SimpleNamespace(
        auto_curriculum=roll_train.POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM,
        best_safe_checkpoint_interval=1,
        best_safe_min_window_episodes=1,
        best_safe_restore_cooldown=1,
        best_safe_restore_mode="weights-only",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _curriculum_log(**overrides: float) -> dict[str, float]:
    prefix = f"Curriculum/{roll_train.POST_C3R_SETTLE_SATURATION_AUTO_CURRICULUM}"
    values = {
        "safe": 1.0,
        "phase_cap_pending": 0.0,
        "window_episodes": 512.0,
        "rollback_count": 0.0,
        "phase_index": 2.0,
        "progress": 0.35,
        "success_rate": 0.99,
        "target_reached_rate": 1.0,
    }
    values.update(overrides)
    return {f"{prefix}/{key}": value for key, value in values.items()}


def test_best_safe_checkpoint_hook_saves_safe_curriculum_state(tmp_path) -> None:
    events: list[tuple[str, str]] = []
    env = SimpleNamespace(
        extras={"log": _curriculum_log()},
        unwrapped=SimpleNamespace(common_step_counter=10),
    )

    def save(path: str) -> None:
        Path(path).write_text("checkpoint", encoding="utf-8")
        events.append(("save", Path(path).name))

    runner = SimpleNamespace(
        logger=SimpleNamespace(log=lambda **_: "logged"),
        save=save,
        load=lambda *_args, **_kwargs: events.append(("load", "unexpected")),
    )

    checkpoint_path = roll_train._install_best_safe_checkpoint_hook(
        runner,
        env=env,
        log_dir=tmp_path,
        args=_best_safe_args(),
        device="cpu",
    )

    assert runner.logger.log(it=7) == "logged"
    assert checkpoint_path == tmp_path / "best_safe.pt"
    assert checkpoint_path.read_text(encoding="utf-8") == "checkpoint"
    metadata = (tmp_path / "best_safe.json").read_text(encoding="utf-8")
    assert '"iteration": 7' in metadata
    assert '"success_rate": 0.99' in metadata
    assert events == [("save", "best_safe.pt")]


def test_best_safe_checkpoint_hook_restores_after_rollback(tmp_path) -> None:
    loads: list[dict[str, object]] = []
    env = SimpleNamespace(
        extras={
            "log": _curriculum_log(
                safe=0.0,
                rollback_count=1.0,
                success_rate=0.94,
            )
        },
        unwrapped=SimpleNamespace(common_step_counter=27),
    )
    (tmp_path / "best_safe.pt").write_text("checkpoint", encoding="utf-8")

    def load(path: str, **kwargs: object) -> None:
        loads.append({"path": Path(path).name, **kwargs})

    runner = SimpleNamespace(
        logger=SimpleNamespace(log=lambda **_: "logged"),
        save=lambda _path: None,
        load=load,
    )

    roll_train._install_best_safe_checkpoint_hook(
        runner,
        env=env,
        log_dir=tmp_path,
        args=_best_safe_args(),
        device="cpu",
    )

    assert runner.logger.log(it=11) == "logged"
    assert len(loads) == 1
    assert loads[0]["path"] == "best_safe.pt"
    assert loads[0]["map_location"] == "cpu"
    assert loads[0]["load_cfg"] == {
        "actor": True,
        "critic": True,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
    }
    assert env.unwrapped.common_step_counter == 0


def test_eval_sync_status_round_trip(tmp_path) -> None:
    path = roll_train._eval_sync_status_path(tmp_path, 49)

    roll_train._write_eval_sync_status(path, None)
    status = roll_train._wait_for_eval_sync_status(
        path,
        poll_interval_s=0.0,
        timeout_s=0.1,
    )

    assert status == {"error": None, "ok": True}


def test_distributed_eval_hook_publishes_status_without_long_collective(
    monkeypatch,
    tmp_path,
) -> None:
    events: list[str] = []
    runner = SimpleNamespace(
        gpu_global_rank=0,
        is_distributed=True,
        logger=SimpleNamespace(log=lambda **_: "logged", writer=None),
        save=lambda _: events.append("save"),
    )

    monkeypatch.setattr(
        roll_train.torch.distributed,
        "barrier",
        lambda: events.append("barrier"),
    )
    monkeypatch.setattr(
        roll_train.torch.distributed,
        "broadcast",
        lambda *_args, **_kwargs: events.append("broadcast"),
    )
    monkeypatch.setattr(
        roll_train,
        "_run_checkpoint_eval",
        lambda **_kwargs: events.append("eval"),
    )

    roll_train._install_eval_hook(
        runner,
        env=SimpleNamespace(),
        log_dir=tmp_path,
        args=_eval_args(),
        device="cpu",
    )
    assert runner.logger.log(it=0) == "logged"

    status = roll_train._wait_for_eval_sync_status(
        roll_train._eval_sync_status_path(tmp_path, 0),
        poll_interval_s=0.0,
        timeout_s=0.1,
    )
    assert status == {"error": None, "ok": True}
    assert events == ["barrier", "save", "eval"]


def test_eval_summary_logging_includes_outcome_and_quality_scalars(tmp_path) -> None:
    scalars: dict[str, float] = {}

    class _Writer:
        def add_scalar(self, tag: str, value: float, iteration: int) -> None:
            assert iteration == 49
            scalars[tag] = value

    summary = {
        "done_rate": 1.0,
        "outcome": {
            "success_rate": 0.75,
            "task_success_rate": 0.75,
            "target_reached_rate": 1.0,
            "excess_pitch_rate": 0.1,
            "excess_depth_error_rate": 0.05,
            "excess_xy_drift_rate": 0.0,
            "time_out_rate": 0.1,
        },
        "first_done_time_s": {"mean": 11.0},
        "terminal": {
            "final_xy_drift_m_peak": {"mean": 0.42, "p90": 0.50},
            "final_pitch_abs_peak_rad": {"p90": 1.1},
            "final_yaw_abs_error_rad": {"p90": 0.2},
            "final_root_ang_speed_rad_s": {"p90": 0.3},
            "final_depth_abs_error_m": {"p90": 0.4},
        },
        "trajectory": {
            "xy_drift_m": {"time_mean_of_env_mean": 0.25},
            "root_ang_speed_rad_s": {"time_mean_of_env_mean": 0.2},
            "depth_abs_error_m": {"time_mean_of_env_mean": 0.1},
            "body_wrench_saturation_fraction": {"time_mean_of_env_mean": 0.45},
        },
    }

    roll_train._log_eval_summary_to_training_tb(
        summary,
        tmp_path,
        49,
        _Writer(),
        tag_prefix="EvalInTraining/current",
    )

    assert scalars["EvalInTraining/current/excess_pitch_rate"] == 0.1
    assert scalars["EvalInTraining/current/time_out_rate"] == 0.1
    assert scalars["EvalInTraining/current/target_reached_rate"] == 1.0
    assert scalars["EvalInTraining/current/pitch_abs_peak_p90"] == 1.1
    assert scalars["EvalInTraining/current/root_ang_speed_time_mean"] == 0.2
