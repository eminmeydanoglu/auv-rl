"""Smoke checks for Taluy roll task."""

from __future__ import annotations

from pathlib import Path

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. Ensure mjlab is available "
        "(local ../mjlab or installed dependency)."
    ) from exc

from auvrl import make_taluy_roll_env_cfg  # noqa: E402  # type: ignore[import-not-found]
from auvrl.tasks.roll.runtime import (  # noqa: E402  # type: ignore[import-not-found]
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
    update_phi_total,
)


EXPECTED_ACTOR_OBS_DIM = 15
EXPECTED_CRITIC_OBS_DIM = 20
EXPECTED_REWARD_TERMS = {
    "roll_progress",
    "xy_drift",
    "pitch_penalty",
    "yaw_hold",
    "depth_hold",
    "action_smoothness",
    "terminal_success",
    "terminal_failure",
}
EXPECTED_TERMINATION_TERMS = {
    "time_out",
    "nan_detected",
    "excess_pitch",
    "excess_depth_error",
    "excess_xy_drift",
    "task_success",
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env() -> ManagerBasedRlEnv:
    cfg = make_taluy_roll_env_cfg(num_envs=2)
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _check_reset_contract(env: ManagerBasedRlEnv) -> None:
    obs, _ = env.reset()
    actor_obs = obs["actor"]
    critic_obs = obs["critic"]

    if actor_obs.shape != (env.num_envs, EXPECTED_ACTOR_OBS_DIM):
        raise AssertionError(
            f"Actor observation has unexpected shape: {tuple(actor_obs.shape)}"
        )
    if critic_obs.shape != (env.num_envs, EXPECTED_CRITIC_OBS_DIM):
        raise AssertionError(
            f"Critic observation has unexpected shape: {tuple(critic_obs.shape)}"
        )

    reward_terms = set(env.reward_manager.active_terms)
    if reward_terms != EXPECTED_REWARD_TERMS:
        raise AssertionError(f"Unexpected reward terms: {sorted(reward_terms)}")

    termination_terms = set(env.termination_manager.active_terms)
    if termination_terms != EXPECTED_TERMINATION_TERMS:
        raise AssertionError(
            f"Unexpected termination terms: {sorted(termination_terms)}"
        )

    state = get_roll_task_state(env)
    if not torch.allclose(state.phi_total_rad, torch.zeros_like(state.phi_total_rad)):
        raise AssertionError("phi_total_rad should reset to zero.")
    if torch.any(state.target_reached):
        raise AssertionError("target_reached should reset to False.")
    if not torch.equal(
        state.settle_counter_steps,
        torch.zeros_like(state.settle_counter_steps),
    ):
        raise AssertionError("settle_counter_steps should reset to zero.")

    last_action_slice = actor_obs[:, 9:15]
    if not torch.allclose(last_action_slice, torch.zeros_like(last_action_slice)):
        raise AssertionError("Actor last-action observation should start at zero.")


def _check_phi_total_tracking(env: ManagerBasedRlEnv) -> None:
    env.reset()

    robot = env.scene["robot"]
    initial_roll, _pitch, _yaw = quat_wxyz_to_roll_pitch_yaw(robot.data.root_link_quat_w)
    manual_phi_total = torch.zeros(env.num_envs, device=env.device)
    prev_roll = initial_roll.clone()

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    action[:, 3] = 0.20

    for _ in range(8):
        _obs, _reward, terminated, truncated, _extras = env.step(action)
        if torch.any(terminated | truncated):
            raise AssertionError("Roll smoke rollout terminated unexpectedly.")

        current_roll, _pitch, _yaw = quat_wxyz_to_roll_pitch_yaw(
            robot.data.root_link_quat_w
        )
        manual_phi_total, prev_roll, _delta_roll = update_phi_total(
            manual_phi_total,
            prev_roll,
            current_roll,
        )
        state = get_roll_task_state(env)
        if not torch.allclose(
            state.phi_total_rad,
            manual_phi_total,
            atol=1.0e-5,
            rtol=1.0e-5,
        ):
            raise AssertionError(
                "Runtime phi_total_rad does not match manual wrapped integration."
            )


def main() -> None:
    env = _make_env()
    try:
        _check_reset_contract(env)
        _check_phi_total_tracking(env)
    finally:
        env.close()

    print("Taluy MJLab roll-task smoke passed.")


if __name__ == "__main__":
    main()
