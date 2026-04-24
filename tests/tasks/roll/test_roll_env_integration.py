from __future__ import annotations

import math

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


def test_roll_smoke_script_runs() -> None:
    roll_smoke.main()


def test_velocity_smoke_regression_runs() -> None:
    velocity_smoke.main()
