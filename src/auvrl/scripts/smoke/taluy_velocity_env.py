"""Smoke checks for the first Taluy velocity-task scaffold."""

from __future__ import annotations

from pathlib import Path
from typing import cast

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer, NullDebugVisualizer
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab. Ensure mjlab is available "
        "(local ../mjlab or installed dependency)."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    UniformBodyVelocityCommand,
    make_taluy_velocity_env_cfg,
)


EXPECTED_COMMAND = torch.tensor([0.25, -0.15, 0.10, 0.30, -0.20, 0.45])
EXPECTED_ACTOR_OBS_DIM = 29
EXPECTED_CRITIC_OBS_DIM = 34
EXPECTED_REWARD_TERMS = {
    "track_linear_velocity",
    "track_angular_velocity",
    "action_l2",
    "action_rate_l2",
    "saturation",
}


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _make_env() -> ManagerBasedRlEnv:
    cfg = make_taluy_velocity_env_cfg(
        num_envs=2,
        command_resampling_time_s=(0.04, 0.04),
        command_rel_zero_envs=0.0,
        command_lin_vel_x_range_m_s=(0.25, 0.25),
        command_lin_vel_y_range_m_s=(-0.15, -0.15),
        command_lin_vel_z_range_m_s=(0.10, 0.10),
        command_ang_vel_x_range_rad_s=(0.30, 0.30),
        command_ang_vel_y_range_rad_s=(-0.20, -0.20),
        command_ang_vel_z_range_rad_s=(0.45, 0.45),
    )
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _check_command_and_observations(env: ManagerBasedRlEnv) -> None:
    obs, _ = env.reset()

    if set(obs.keys()) != {"actor", "critic"}:
        raise AssertionError(f"Unexpected observation groups: {sorted(obs.keys())}")

    actor_obs = obs["actor"]
    critic_obs = obs["critic"]
    if not isinstance(actor_obs, torch.Tensor) or not isinstance(
        critic_obs, torch.Tensor
    ):
        raise AssertionError("Actor/critic observations should be tensors.")
    if actor_obs.shape != (env.num_envs, EXPECTED_ACTOR_OBS_DIM):
        raise AssertionError(
            f"Actor observation has unexpected shape. Observed {tuple(actor_obs.shape)}"
        )
    if critic_obs.shape != (env.num_envs, EXPECTED_CRITIC_OBS_DIM):
        raise AssertionError(
            "Critic observation has unexpected shape. "
            f"Observed {tuple(critic_obs.shape)}"
        )

    command_term = env.command_manager.get_term("body_velocity")
    if not isinstance(command_term, UniformBodyVelocityCommand):
        raise AssertionError(
            "Expected body_velocity term to be UniformBodyVelocityCommand"
        )

    expected_command = (
        EXPECTED_COMMAND.to(device=env.device).view(1, 6).expand(env.num_envs, -1)
    )
    command = command_term.command
    if command.shape != (env.num_envs, 6):
        raise AssertionError(f"Unexpected command shape: {tuple(command.shape)}")
    if not torch.allclose(command, expected_command):
        raise AssertionError(
            "Command did not match fixed test value. "
            f"Observed {command.detach().cpu().tolist()}"
        )

    command_scale = torch.tensor(
        [4.0, 6.6666665, 10.0, 3.3333333, 5.0, 2.2222223],
        device=env.device,
    ).view(1, 6)
    command_slice = actor_obs[:, 9:15]
    expected_command_obs = expected_command * command_scale
    if not torch.allclose(command_slice, expected_command_obs.expand_as(command_slice)):
        raise AssertionError(
            "Actor observation command slice does not match command manager output."
        )

    last_action_slice = actor_obs[:, 15:21]
    if not torch.allclose(last_action_slice, torch.zeros_like(last_action_slice)):
        raise AssertionError("Last-action slice should start at zero after reset.")

    thruster_force_slice = actor_obs[:, 21:29]
    if not torch.allclose(
        thruster_force_slice,
        torch.zeros_like(thruster_force_slice),
    ):
        raise AssertionError("Thruster-force slice should start at zero after reset.")

    critic_depth = critic_obs[:, 29:30]
    critic_current = critic_obs[:, 30:33]
    critic_voltage = critic_obs[:, 33:34]
    if not torch.allclose(critic_depth, torch.zeros_like(critic_depth)):
        raise AssertionError(
            "Critic depth-error term should start at zero after reset."
        )
    if not torch.allclose(critic_current, torch.zeros_like(critic_current)):
        raise AssertionError(
            "Critic current-velocity term should start at zero with no current."
        )
    if not torch.allclose(critic_voltage, torch.zeros_like(critic_voltage)):
        raise AssertionError(
            "Critic voltage-offset term should start at zero at nominal voltage."
        )

    if int(command_term.command_counter.min().item()) != 1:
        raise AssertionError(
            "Command term should have sampled once during reset. "
            f"Observed counters {command_term.command_counter.tolist()}"
        )

    reward_terms = set(env.reward_manager.active_terms)
    if reward_terms != EXPECTED_REWARD_TERMS:
        raise AssertionError(f"Unexpected reward terms: {sorted(reward_terms)}")


def _check_resampling_and_debug_vis(env: ManagerBasedRlEnv) -> None:
    command_term = env.command_manager.get_term("body_velocity")
    assert isinstance(command_term, UniformBodyVelocityCommand)

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    initial_counter = command_term.command_counter.clone()

    for _ in range(6):
        _, reward, _, _, _ = env.step(action)
        _assert_finite_env_state(env)
        if not torch.isfinite(reward).all():
            raise AssertionError(f"Encountered non-finite rewards: {reward.tolist()}")

    if not torch.all(command_term.command_counter >= initial_counter + 1):
        raise AssertionError(
            "Command did not resample after the configured interval. "
            f"Initial={initial_counter.tolist()} current={command_term.command_counter.tolist()}"
        )

    visualizer = cast(DebugVisualizer, NullDebugVisualizer(env_idx=0, meansize=0.5))
    command_term.debug_vis(visualizer)


def main() -> None:
    env = _make_env()
    try:
        _check_command_and_observations(env)
        _check_resampling_and_debug_vis(env)
    finally:
        env.close()

    print("Taluy MJLab velocity-task scaffold smoke passed.")


if __name__ == "__main__":
    main()
