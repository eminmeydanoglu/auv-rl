from __future__ import annotations

import argparse
import math
from typing import Any, cast

from mjlab.envs import ManagerBasedRlEnv

from auvrl import (
    ROLL_CURRICULUM_STAGES,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.scripts.demo import taluy_roll_play
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


def test_roll_play_inspector_env_cfg_accepts_curriculum_stage() -> None:
    stage = ROLL_CURRICULUM_STAGES["c0_90_discovery"]
    args = argparse.Namespace(
        num_envs=1,
        curriculum_stage=stage.name,
        episode_length_s=None,
        no_terminations=False,
    )

    cfg = taluy_roll_play._make_roll_inspector_env_cfg(args)

    assert cfg.episode_length_s == stage.episode_length_s
    assert cfg.rewards["roll_progress"].weight == stage.k_prog
    assert cfg.terminations["task_success"].params["target_roll_rad"] == math.radians(
        stage.target_roll_deg
    )
    assert "body_velocity" in cfg.commands


def test_roll_play_checkpoint_helpers_prefer_direct_path(tmp_path) -> None:
    checkpoint_path = tmp_path / "model_0.pt"
    checkpoint_path.write_bytes(b"placeholder")
    args = argparse.Namespace(
        policy="manual",
        checkpoint_file=checkpoint_path,
        experiment_name="ignored",
        run_dir="ignored",
        checkpoint="ignored",
    )

    assert taluy_roll_play._checkpoint_lookup_requested(args)
    assert taluy_roll_play._resolve_checkpoint_path(args) == checkpoint_path.resolve()
    assert (
        taluy_roll_play._load_agent_cfg_dict(checkpoint_path)["experiment_name"]
        == taluy_roll_ppo_runner_cfg().experiment_name
    )


def test_switchable_roll_policy_uses_selected_callable() -> None:
    import torch

    def manual_policy(_obs):
        return torch.zeros((1, 6))

    def checkpoint_policy(_obs):
        return torch.ones((1, 6))

    policy = taluy_roll_play.SwitchableRollPolicy(
        manual_policy=manual_policy,
        checkpoint_policy=checkpoint_policy,
        mode="manual",
        checkpoint_path=None,
    )

    assert policy.mode == "manual"
    assert torch.equal(policy(object()), torch.zeros((1, 6)))
    policy.set_mode("checkpoint")
    assert policy.mode == "checkpoint"
    assert torch.equal(policy(object()), torch.ones((1, 6)))


def test_roll_smoke_script_runs() -> None:
    roll_smoke.main()


def test_velocity_smoke_regression_runs() -> None:
    velocity_smoke.main()
