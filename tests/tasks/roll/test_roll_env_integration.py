from __future__ import annotations

import math
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnv

from auvrl import (
    ROLL_CURRICULUM_STAGES,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.scripts.smoke import taluy_roll_env as roll_smoke
from auvrl.scripts.smoke import taluy_velocity_env as velocity_smoke
from auvrl.tasks.roll.runtime import get_roll_task_state


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def test_roll_env_cfg_api_and_reset_state() -> None:
    cfg = make_taluy_roll_env_cfg(num_envs=2)
    assert cfg.episode_length_s == 20.0
    assert set(cfg.rewards.keys()) == {
        "roll_progress",
        "xy_drift",
        "pitch_penalty",
        "yaw_hold",
        "depth_hold",
        "action_smoothness",
        "terminal_success",
        "terminal_failure",
    }
    assert set(cfg.metrics.keys()) == {
        "roll_progress_ratio_last",
        "phi_total_rad_last",
        "target_reached_last",
        "settle_counter_s_last",
        "depth_abs_error_m",
        "xy_drift_m",
        "xy_drift_m_last",
        "x_drift_m_last",
        "y_drift_m_last",
        "xy_drift_m_peak",
        "pitch_abs_rad",
        "yaw_abs_error_rad",
        "root_ang_speed_rad_s",
        "body_wrench_action_l2",
        "body_wrench_saturation_fraction",
        "water_current_speed_m_s",
        "hydro_wrench_norm",
    }
    assert set(cfg.terminations.keys()) == {
        "time_out",
        "nan_detected",
        "excess_pitch",
        "excess_depth_error",
        "excess_xy_drift",
        "task_success",
    }
    assert "reset_roll_task_state" in cfg.events

    env = ManagerBasedRlEnv(cfg=cfg, device=_device())
    try:
        obs, _ = env.reset()
        obs = cast(dict[str, Any], obs)
        assert obs["actor"].shape == (2, 15)
        assert obs["critic"].shape == (2, 20)

        state = get_roll_task_state(env)
        assert state.phi_total_rad.shape == (2,)
        assert not state.target_reached.any().item()
        assert state.settle_counter_steps.eq(0).all().item()
    finally:
        env.close()


def test_roll_ppo_runner_cfg_uses_longer_rollout_horizon() -> None:
    cfg = taluy_roll_ppo_runner_cfg()
    assert cfg.num_steps_per_env == 256
    assert cfg.actor.hidden_dims == (512, 256, 128)
    assert cfg.critic.hidden_dims == (512, 256, 128)


def test_roll_curriculum_c0_applies_static_stage_params() -> None:
    stage = ROLL_CURRICULUM_STAGES["c0_90_discovery"]
    cfg = make_taluy_roll_env_cfg(num_envs=1, curriculum_stage=stage.name)

    assert cfg.episode_length_s == stage.episode_length_s
    assert cfg.rewards["xy_drift"].weight == 0.0
    assert cfg.rewards["roll_progress"].weight == stage.k_prog
    assert cfg.rewards["terminal_success"].weight == stage.terminal_success_weight
    assert cfg.terminations["excess_xy_drift"].params["limit_m"] == (
        stage.excess_xy_drift_m
    )
    assert cfg.terminations["task_success"].params["target_roll_rad"] == math.radians(
        stage.target_roll_deg
    )
    assert cfg.observations["actor"].terms["phi_total_norm"].params[
        "target_roll_rad"
    ] == math.radians(stage.target_roll_deg)
    assert cfg.rewards["roll_progress"].params["target_roll_rad"] == math.radians(
        stage.target_roll_deg
    )


def test_roll_curriculum_c2a_reach_stage_adds_attitude_discipline() -> None:
    stage = ROLL_CURRICULUM_STAGES["c2a_360_reach"]
    cfg = make_taluy_roll_env_cfg(num_envs=1, curriculum_stage=stage.name)

    assert cfg.episode_length_s == 14.0
    assert cfg.rewards["roll_progress"].weight == 6.0
    assert cfg.rewards["xy_drift"].weight == 0.02
    assert cfg.rewards["pitch_penalty"].weight == 1.5
    assert cfg.rewards["yaw_hold"].weight == 0.6
    assert cfg.rewards["depth_hold"].weight == 0.5
    assert cfg.rewards["terminal_success"].weight == 150.0
    assert cfg.rewards["terminal_failure"].weight == -40.0
    assert cfg.terminations["excess_pitch"].params["limit_rad"] == math.radians(70.0)
    assert cfg.terminations["excess_xy_drift"].params["limit_m"] == 6.0
    assert cfg.terminations["excess_depth_error"].params["limit_m"] == 1.8
    assert cfg.terminations["task_success"].params["settle_steps"] == 3
    assert cfg.terminations["task_success"].params[
        "settle_ang_vel_limit_rad_s"
    ] == 2.5


def test_roll_curriculum_next_wave_matches_training_plan() -> None:
    expected = {
        "c2b_360_hold_0p05": {
            "target_deg": 360.0,
            "episode_s": 14.0,
            "settle_steps": 7,
            "k_prog": 6.0,
            "k_pitch": 1.5,
            "terminal_success": 170.0,
            "excess_pitch_deg": 70.0,
            "excess_xy": 6.0,
        },
        "c2c_360_hold_0p10": {
            "target_deg": 360.0,
            "episode_s": 14.0,
            "settle_steps": 13,
            "k_prog": 6.0,
            "k_pitch": 1.5,
            "terminal_success": 180.0,
            "excess_pitch_deg": 70.0,
            "excess_xy": 6.0,
        },
        "c3a_540_reach_0p05": {
            "target_deg": 540.0,
            "episode_s": 17.0,
            "settle_steps": 7,
            "k_prog": 6.0,
            "k_pitch": 1.2,
            "terminal_success": 170.0,
            "excess_pitch_deg": 75.0,
            "excess_xy": 7.0,
        },
        "c3b_720_reach_0p02": {
            "target_deg": 720.0,
            "episode_s": 20.0,
            "settle_steps": 3,
            "k_prog": 6.0,
            "k_pitch": 1.0,
            "terminal_success": 170.0,
            "excess_pitch_deg": 80.0,
            "excess_xy": 8.0,
        },
    }

    for stage_name, values in expected.items():
        stage = ROLL_CURRICULUM_STAGES[stage_name]
        cfg = make_taluy_roll_env_cfg(num_envs=1, curriculum_stage=stage.name)

        assert cfg.episode_length_s == values["episode_s"]
        assert cfg.rewards["roll_progress"].weight == values["k_prog"]
        assert cfg.rewards["pitch_penalty"].weight == values["k_pitch"]
        assert cfg.rewards["terminal_success"].weight == values["terminal_success"]
        assert cfg.terminations["excess_pitch"].params["limit_rad"] == math.radians(
            values["excess_pitch_deg"]
        )
        assert cfg.terminations["excess_xy_drift"].params["limit_m"] == values[
            "excess_xy"
        ]
        assert cfg.terminations["task_success"].params["target_roll_rad"] == math.radians(
            values["target_deg"]
        )
        assert cfg.terminations["task_success"].params["settle_steps"] == values[
            "settle_steps"
        ]
        assert cfg.observations["actor"].terms["phi_total_norm"].params[
            "target_roll_rad"
        ] == math.radians(values["target_deg"])


def test_roll_curriculum_720_wave_matches_direct_probe_plan() -> None:
    expected = {
        "c3c_720_reach_loose_control": {
            "settle_steps": 3,
            "k_xy": 0.02,
            "terminal_success": 170.0,
        },
        "c3d_720_reach_xy_moderate": {
            "settle_steps": 3,
            "k_xy": 0.06,
            "terminal_success": 170.0,
        },
        "c3e_720_hold_0p05_xy_light": {
            "settle_steps": 7,
            "k_xy": 0.04,
            "terminal_success": 180.0,
        },
        "c3f_720_hold_0p10_soft": {
            "settle_steps": 13,
            "k_xy": 0.05,
            "terminal_success": 190.0,
        },
    }

    for stage_name, values in expected.items():
        cfg = make_taluy_roll_env_cfg(num_envs=1, curriculum_stage=stage_name)

        assert cfg.episode_length_s == 20.0
        assert cfg.rewards["roll_progress"].weight == 6.0
        assert cfg.rewards["xy_drift"].weight == values["k_xy"]
        assert cfg.rewards["terminal_success"].weight == values["terminal_success"]
        assert cfg.terminations["task_success"].params["target_roll_rad"] == math.radians(
            720.0
        )
        assert cfg.terminations["task_success"].params["settle_steps"] == values[
            "settle_steps"
        ]
        assert cfg.terminations["task_success"].params[
            "settle_ang_vel_limit_rad_s"
        ] == 3.5
        assert cfg.terminations["excess_xy_drift"].params["limit_m"] == 8.0


def test_roll_curriculum_low_saturation_experiments_match_plan() -> None:
    expected_sat_weights = {
        "c3i_720_hold_0p10_sat005": 0.05,
        "c3i_720_hold_0p10_sat010": 0.10,
        "c3i_720_hold_0p10_sat020": 0.20,
    }

    for stage_name, sat_weight in expected_sat_weights.items():
        stage = ROLL_CURRICULUM_STAGES[stage_name]
        cfg = make_taluy_roll_env_cfg(num_envs=1, curriculum_stage=stage_name)

        assert cfg.episode_length_s == 20.0
        assert cfg.rewards["roll_progress"].weight == 6.0
        assert cfg.rewards["xy_drift"].weight == 0.15
        assert cfg.rewards["action_smoothness"].weight == -0.010
        assert cfg.rewards["action_effort"].weight == -0.003
        assert cfg.rewards["thruster_saturation"].weight == -sat_weight
        assert cfg.rewards["thruster_saturation"].params["threshold"] == 0.85
        assert stage.k_thruster_saturation == sat_weight
        assert cfg.terminations["task_success"].params["target_roll_rad"] == math.radians(
            720.0
        )
        assert cfg.terminations["task_success"].params["settle_steps"] == 13
        assert cfg.terminations["task_success"].params[
            "settle_ang_vel_limit_rad_s"
        ] == 2.0
        assert cfg.terminations["excess_xy_drift"].params["limit_m"] == 2.5


def test_roll_curriculum_xy_tight_success_requires_lower_settle_drift() -> None:
    cfg = make_taluy_roll_env_cfg(
        num_envs=1,
        curriculum_stage="c3h_720_hold_0p05_xy_tight",
    )

    success_params = cfg.terminations["task_success"].params
    assert cfg.terminations["excess_xy_drift"].params["limit_m"] == 2.5
    assert success_params["settle_xy_drift_limit_m"] == 1.25


def test_roll_smoke_script_runs() -> None:
    roll_smoke.main()


def test_velocity_smoke_regression_runs() -> None:
    velocity_smoke.main()
