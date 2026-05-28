from __future__ import annotations

from types import SimpleNamespace

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from auvrl.tasks.roll.auto_curriculum import (
    PostC3LPolishCurriculum,
    PostC3LPolishSchedule,
)
from auvrl.tasks.roll.curriculum import get_roll_curriculum_stage


class _RewardManager:
    def __init__(self) -> None:
        self._terms = {
            "xy_drift": SimpleNamespace(weight=0.28, params={}),
            "pitch_penalty": SimpleNamespace(weight=1.0, params={}),
            "yaw_hold": SimpleNamespace(weight=0.4, params={}),
            "depth_hold": SimpleNamespace(weight=0.55, params={}),
            "action_smoothness": SimpleNamespace(weight=-0.010, params={}),
            "action_effort": SimpleNamespace(weight=-0.003, params={}),
            "thruster_saturation": SimpleNamespace(
                weight=-0.10,
                params={"threshold": 0.85},
            ),
        }

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        return self._terms[name]


class _TerminationManager:
    def __init__(self, success: torch.Tensor) -> None:
        self.success = success
        self.terms = {
            "task_success": success,
            "time_out": torch.zeros_like(success),
            "excess_pitch": torch.zeros_like(success),
            "excess_depth_error": torch.zeros_like(success),
            "excess_xy_drift": torch.zeros_like(success),
        }
        self.task_success_cfg = SimpleNamespace(
            params={
                "settle_steps": 13,
                "settle_pitch_limit_rad": torch.pi * 45.0 / 180.0,
                "settle_yaw_limit_rad": torch.pi * 90.0 / 180.0,
                "settle_ang_vel_limit_rad_s": 2.0,
                "settle_depth_error_limit_m": 1.5,
                "settle_xy_drift_limit_m": 0.9,
            }
        )
        self.excess_pitch_cfg = SimpleNamespace(
            params={
                "limit_rad": torch.pi * 80.0 / 180.0,
            }
        )

    def get_term(self, name: str) -> torch.Tensor:
        if name == "task_success":
            self.terms[name] = self.success
        return self.terms[name]

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        if name == "task_success":
            return self.task_success_cfg
        if name == "excess_pitch":
            return self.excess_pitch_cfg
        raise KeyError(name)


class _MetricsManager:
    def __init__(self) -> None:
        self.active_terms = [
            "xy_drift_m_peak",
            "pitch_abs_peak_rad",
            "depth_abs_error_m",
            "pitch_abs_rad",
            "yaw_abs_error_rad",
            "root_ang_speed_rad_s",
            "body_wrench_action_l2",
            "body_wrench_saturation_fraction",
            "target_reached_last",
        ]
        self._term_cfgs = [
            SimpleNamespace(reduce="last"),
            SimpleNamespace(reduce="last"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="last"),
        ]
        self._step_count = torch.tensor([920, 920], dtype=torch.long)
        self._step_values = torch.tensor(
            [
                [0.20, 0.30, 0.10, 0.20, 0.40, 1.0, 4.0, 0.50],
                [0.22, 0.32, 0.12, 0.22, 0.42, 1.1, 4.2, 0.52],
            ],
            dtype=torch.float,
        )
        self._step_values = torch.cat(
            [self._step_values, torch.ones((2, 1), dtype=torch.float)],
            dim=1,
        )
        self._episode_sums = {
            name: self._step_values[:, index] * self._step_count.float()
            for index, name in enumerate(self.active_terms)
        }


class _Env:
    def __init__(self) -> None:
        self.num_envs = 2
        self.device = "cpu"
        self.step_dt = 0.01
        self.reward_manager = _RewardManager()
        self.termination_manager = _TerminationManager(
            torch.tensor([True, True], dtype=torch.bool)
        )
        self.metrics_manager = _MetricsManager()


def _term() -> tuple[PostC3LPolishCurriculum, _Env, PostC3LPolishSchedule]:
    schedule = PostC3LPolishSchedule(
        start_stage=get_roll_curriculum_stage("c3l_720_xy_guard"),
        goal_stage=get_roll_curriculum_stage("c3q_720_c3l_strict_settle"),
        rolling_window_episodes=2,
        min_completed_episodes_per_update=2,
        observe_updates=1,
    )
    cfg = CurriculumTermCfg(
        func=PostC3LPolishCurriculum,
        params={"schedule": schedule},
    )
    env = _Env()
    return PostC3LPolishCurriculum(cfg=cfg, env=env), env, schedule


def test_post_c3l_polish_advances_attitude_depth_after_safe_windows() -> None:
    term, env, schedule = _term()
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    first_state = term(env, env_ids, schedule=schedule)
    assert first_state["phase_index"] == 1.0
    assert first_state["k_pitch"] == 1.0

    second_state = term(env, env_ids, schedule=schedule)
    assert second_state["phase_index"] == 1.0
    assert second_state["last_update"] == 1.0
    assert second_state["k_xy"] == 0.28500000000000003
    assert second_state["k_pitch"] == 1.025
    assert second_state["k_yaw"] == 0.41000000000000003
    assert second_state["k_depth"] == 0.56
    assert env.reward_manager.get_term_cfg("xy_drift").weight == second_state["k_xy"]
    assert env.reward_manager.get_term_cfg("pitch_penalty").weight == second_state[
        "k_pitch"
    ]
    assert second_state["phase_is_attitude_depth"] == 1.0
    assert second_state["advance_blocked_by_success"] == 0.0
    assert second_state["advance_blocked_by_xy"] == 0.0
    assert second_state["advance_blocked_by_pitch"] == 0.0
    assert second_state["target_reached_rate"] == 1.0
    assert second_state["reached_but_not_success_rate"] == 0.0
    assert second_state["time_out_rate"] == 0.0
    assert second_state["excess_pitch_rate"] == 0.0
    assert second_state["settle_all_ok_rate"] == 1.0
    assert second_state["phase_update_count"] == 1.0
    assert second_state["phase_completed_episodes"] == 2.0
    assert second_state["phase_advance_count"] == 1.0


def test_post_c3l_polish_splits_hard_pitch_from_settle_motion() -> None:
    term, env, schedule = _term()
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    state = term(env, env_ids, schedule=schedule)
    for _ in range(80):
        if state["phase_is_hard_pitch_envelope"] == 1.0:
            break
        state = term(env, env_ids, schedule=schedule)

    assert state["phase_is_hard_pitch_envelope"] == 1.0
    assert state["phase_is_settle_ang_vel"] == 0.0
    assert state["advance_pitch_peak_limit_deg"] == state["excess_pitch_deg"] + 1.0


def test_post_c3l_polish_logs_first_done_without_gating_curriculum() -> None:
    term, _, _ = _term()
    term._success.extend([1.0, 1.0])
    term._first_done_s.extend([30.0, 30.0])
    term._xy_peak_m.extend([0.20, 0.20])
    term._pitch_peak_rad.extend([0.30, 0.30])
    term._depth_abs_error_m.extend([0.10, 0.10])
    term._root_ang_speed_rad_s.extend([1.0, 1.0])
    term._action_l2.extend([4.0, 4.0])
    term._saturation.extend([0.50, 0.50])

    term._phase_index = term._PHASES.index("settle_window")
    state = term._state()
    assert state["first_done_time_mean_s"] == 30.0
    assert "advance_first_done_limit_s" not in state
    assert "advance_first_done_margin_s" not in state
    assert "advance_blocked_by_first_done" not in state
    assert "rollback_first_done_limit_s" not in state
    assert term._can_advance()

    term._phase_index = term._PHASES.index("settle_ang_vel")
    assert term._can_advance()
    assert not term._must_rollback()


def test_post_c3l_polish_logs_rollback_reasons_and_margins() -> None:
    term, _, _ = _term()
    term._success.extend([0.0, 1.0])
    term._target_reached.extend([1.0, 1.0])
    term._time_out.extend([1.0, 0.0])
    term._excess_pitch.extend([0.0, 0.0])
    term._excess_depth_error.extend([0.0, 0.0])
    term._excess_xy_drift.extend([0.0, 0.0])
    term._xy_peak_m.extend([0.80, 0.75])
    term._pitch_peak_rad.extend([0.30, 0.30])
    term._depth_abs_error_m.extend([0.10, 0.10])
    term._pitch_abs_rad.extend([0.20, 0.20])
    term._yaw_abs_error_rad.extend([0.10, 0.10])
    term._root_ang_speed_rad_s.extend([1.0, 1.0])
    term._action_l2.extend([4.0, 4.0])
    term._saturation.extend([0.50, 0.50])

    state = term._state()

    assert state["success_rate"] == 0.5
    assert state["target_reached_rate"] == 1.0
    assert state["reached_but_not_success_rate"] == 0.5
    assert state["time_out_rate"] == 0.5
    assert state["rollback_blocked_by_success"] == 1.0
    assert state["rollback_blocked_by_xy"] == 1.0
    assert state["rollback_success_margin"] == -0.44999999999999996
    assert state["rollback_xy_peak_margin_m"] < 0.0
    assert term._must_rollback()


def test_post_c3l_polish_rolls_back_xy_before_settle_xy_phase() -> None:
    term, _, schedule = _term()
    term._success.extend([1.0, 1.0])
    term._xy_peak_m.extend([0.824, 0.824])
    term._pitch_peak_rad.extend([0.30, 0.30])
    term._depth_abs_error_m.extend([0.10, 0.10])
    term._root_ang_speed_rad_s.extend([1.0, 1.0])
    term._action_l2.extend([4.0, 4.0])
    term._saturation.extend([0.50, 0.50])
    term._values["settle_xy_drift_limit_m"] = 0.9

    term._phase_index = term._PHASES.index("settle_ang_vel")
    state = term._state()
    assert state["rollback_xy_peak_limit_m"] == schedule.pre_settle_xy_rollback_max_m
    assert state["rollback_blocked_by_xy"] == 1.0
    assert term._must_rollback()

    term._phase_index = term._PHASES.index("settle_xy")
    state = term._state()
    assert state["rollback_xy_peak_limit_m"] == 0.9
    assert state["rollback_blocked_by_xy"] == 0.0
    assert not term._must_rollback()


def test_post_c3l_polish_tolerates_moderate_xy_before_settle_xy_phase() -> None:
    term, _, schedule = _term()
    term._success.extend([1.0, 1.0])
    term._xy_peak_m.extend([0.60, 0.60])
    term._pitch_peak_rad.extend([0.30, 0.30])
    term._depth_abs_error_m.extend([0.10, 0.10])
    term._root_ang_speed_rad_s.extend([1.0, 1.0])
    term._action_l2.extend([4.0, 4.0])
    term._saturation.extend([0.50, 0.50])
    term._values["settle_xy_drift_limit_m"] = 0.9

    term._phase_index = term._PHASES.index("settle_ang_vel")
    state = term._state()
    assert state["rollback_xy_peak_limit_m"] == schedule.pre_settle_xy_rollback_max_m
    assert state["rollback_blocked_by_xy"] == 0.0
    assert not term._must_rollback()


def test_post_c3l_polish_rolls_back_attitude_depth_after_unsafe_window() -> None:
    term, env, schedule = _term()
    env_ids = torch.tensor([0, 1], dtype=torch.long)

    term(env, env_ids, schedule=schedule)
    term(env, env_ids, schedule=schedule)
    env.termination_manager.success = torch.tensor([False, False], dtype=torch.bool)
    rollback_state = term(env, env_ids, schedule=schedule)

    assert rollback_state["last_update"] == -1.0
    assert rollback_state["last_rollback"] == 1.0
    assert rollback_state["k_xy"] == 0.28
    assert rollback_state["k_pitch"] == 1.0
    assert rollback_state["k_yaw"] == 0.4
    assert rollback_state["k_depth"] == 0.55
    assert env.reward_manager.get_term_cfg("xy_drift").weight == 0.28
    assert env.reward_manager.get_term_cfg("pitch_penalty").weight == 1.0
