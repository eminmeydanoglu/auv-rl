from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnv

from auvrl import make_taluy_roll_env_cfg, taluy_roll_ppo_runner_cfg
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


def test_roll_smoke_script_runs() -> None:
    roll_smoke.main()


def test_velocity_smoke_regression_runs() -> None:
    velocity_smoke.main()
