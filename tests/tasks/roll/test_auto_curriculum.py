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

    def get_term(self, name: str) -> torch.Tensor:
        assert name == "task_success"
        return self.success

    def get_term_cfg(self, name: str) -> SimpleNamespace:
        assert name == "task_success"
        return self.task_success_cfg


class _MetricsManager:
    def __init__(self) -> None:
        self.active_terms = [
            "xy_drift_m_peak",
            "depth_abs_error_m",
            "pitch_abs_rad",
            "yaw_abs_error_rad",
            "root_ang_speed_rad_s",
            "body_wrench_action_l2",
            "body_wrench_saturation_fraction",
        ]
        self._term_cfgs = [
            SimpleNamespace(reduce="last"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
            SimpleNamespace(reduce="mean"),
        ]
        self._step_count = torch.tensor([920, 920], dtype=torch.long)
        self._step_values = torch.tensor(
            [
                [0.20, 0.10, 0.20, 0.40, 1.0, 4.0, 0.50],
                [0.22, 0.12, 0.22, 0.42, 1.1, 4.2, 0.52],
            ],
            dtype=torch.float,
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
        goal_stage=get_roll_curriculum_stage("c3p_720_c3l_deploy_polish"),
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
