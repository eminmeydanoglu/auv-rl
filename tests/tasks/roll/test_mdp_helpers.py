from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz

from auvrl.tasks.roll import mdp
from auvrl.tasks.roll.runtime import get_roll_task_state


class DummyTerminationManager:
    def __init__(self, terms: dict[str, torch.Tensor]):
        self._terms = terms

    def get_term(self, name: str) -> torch.Tensor:
        return self._terms[name]


class DummyActionManager:
    def __init__(self, action: torch.Tensor, prev_action: torch.Tensor):
        self.action = action
        self.prev_action = prev_action
        self.active_terms = ["hydro", "body_wrench"]
        self.action_term_dim = [0, action.shape[1]]


class DummyEnv:
    def __init__(
        self,
        *,
        quat_wxyz: torch.Tensor,
        ang_vel_b: torch.Tensor,
        lin_vel_b: torch.Tensor,
        pos_w: torch.Tensor,
        action: torch.Tensor,
        prev_action: torch.Tensor,
    ):
        self.device = str(quat_wxyz.device)
        self.num_envs = quat_wxyz.shape[0]
        self.episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=quat_wxyz.device,
        )
        self.scene = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    root_link_quat_w=quat_wxyz,
                    root_link_ang_vel_b=ang_vel_b,
                    root_link_lin_vel_b=lin_vel_b,
                    root_link_pos_w=pos_w,
                )
            )
        }
        self.action_manager = DummyActionManager(action=action, prev_action=prev_action)
        self.termination_manager = DummyTerminationManager(
            {
                "task_success": torch.zeros(
                    self.num_envs, dtype=torch.bool, device=quat_wxyz.device
                )
            }
        )
        self.reset_buf = torch.zeros(
            self.num_envs, dtype=torch.bool, device=quat_wxyz.device
        )


def _make_env() -> DummyEnv:
    quat = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )
    ang_vel_b = torch.zeros(1, 3, dtype=torch.float)
    lin_vel_b = torch.tensor([[0.1, -0.2, 0.3]], dtype=torch.float)
    pos_w = torch.tensor([[1.0, -2.0, -0.4]], dtype=torch.float)
    action = torch.tensor([[0.0, 0.0, 0.0, 0.3, 0.0, 0.0]], dtype=torch.float)
    prev_action = torch.zeros_like(action)
    return DummyEnv(
        quat_wxyz=quat,
        ang_vel_b=ang_vel_b,
        lin_vel_b=lin_vel_b,
        pos_w=pos_w,
        action=action,
        prev_action=prev_action,
    )


def test_observation_helpers_use_reset_latched_references() -> None:
    env = _make_env()
    state = get_roll_task_state(env)

    assert torch.allclose(state.phi_total_rad, torch.zeros(1))
    assert torch.equal(state.target_reached, torch.tensor([False]))
    assert torch.equal(state.settle_counter_steps, torch.tensor([0]))

    depth_error = mdp.depth_error_from_ref(env)
    xy_error = mdp.xy_error_w(env)
    phi_total = mdp.phi_total_norm(
        env,
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
    )
    last_action = mdp.last_body_wrench_action(env)

    assert torch.allclose(depth_error, torch.zeros(1, 1))
    assert torch.allclose(xy_error, torch.zeros(1, 2))
    assert torch.allclose(phi_total, torch.zeros(1, 1))
    assert torch.allclose(last_action, torch.zeros(1, 6))


def test_roll_progress_is_positive_only_in_commanded_direction() -> None:
    env = _make_env()
    get_roll_task_state(env)

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([0.20], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )
    forward_progress = mdp.roll_progress(
        env,
        roll_direction=1,
        target_roll_rad=math.pi,
    )
    reverse_progress = mdp.roll_progress(
        env,
        roll_direction=-1,
        target_roll_rad=math.pi,
    )

    assert torch.all(forward_progress > 0.0)
    assert torch.allclose(reverse_progress, torch.zeros_like(reverse_progress))


def test_roll_progress_is_capped_at_target() -> None:
    env = _make_env()
    state = get_roll_task_state(env)
    state.phi_total_rad[:] = math.radians(350.0)
    state.prev_roll_rad[:] = 0.0

    env.episode_length_buf[:] = 1
    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([math.radians(20.0)], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )

    progress = mdp.roll_progress(
        env,
        roll_direction=1,
        target_roll_rad=math.radians(360.0),
    )

    assert torch.allclose(progress, torch.tensor([10.0 / 360.0]), atol=1.0e-6)

    env.episode_length_buf[:] = 2
    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([math.radians(40.0)], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )

    post_target_progress = mdp.roll_progress(
        env,
        roll_direction=1,
        target_roll_rad=math.radians(360.0),
    )

    assert torch.allclose(post_target_progress, torch.zeros_like(post_target_progress))


def test_action_rate_reward_matches_l2_delta() -> None:
    env = _make_env()
    value = mdp.body_wrench_action_rate_l2(env)
    assert torch.allclose(value, torch.tensor([0.09]))


def test_terminal_rewards_read_termination_results() -> None:
    env = _make_env()
    env.termination_manager = DummyTerminationManager(
        {"task_success": torch.tensor([True])}
    )
    env.reset_buf = torch.tensor([True])

    success = mdp.terminal_success_reward(env)
    failure = mdp.terminal_failure_reward(env)

    assert torch.equal(success, torch.tensor([1.0]))
    assert torch.equal(failure, torch.tensor([0.0]))

    env.termination_manager = DummyTerminationManager(
        {"task_success": torch.tensor([False])}
    )
    success = mdp.terminal_success_reward(env)
    failure = mdp.terminal_failure_reward(env)

    assert torch.equal(success, torch.tensor([0.0]))
    assert torch.equal(failure, torch.tensor([1.0]))


def test_termination_helpers_and_success_logic() -> None:
    env = _make_env()
    get_roll_task_state(env)
    state = get_roll_task_state(env)
    state.phi_total_rad[:] = 4.0 * math.pi
    env.episode_length_buf[:] = 1

    success = mdp.roll_task_success(
        env,
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
        settle_steps=1,
    )
    assert torch.equal(success, torch.tensor([True]))
    assert torch.equal(state.target_reached, torch.tensor([True]))
    assert torch.equal(state.settle_counter_steps, torch.tensor([1]))

    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([math.radians(85.0)], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )
    assert torch.equal(mdp.excess_pitch(env), torch.tensor([True]))

    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
        torch.tensor([0.0], dtype=torch.float),
    )
    env.scene["robot"].data.root_link_pos_w[:, 2] = state.z_ref_m + 1.2
    assert torch.equal(mdp.excess_depth_error(env), torch.tensor([True]))

    env.scene["robot"].data.root_link_pos_w[:, :2] = state.xy_ref_w + torch.tensor(
        [[1.2, 0.0]],
        dtype=torch.float,
    )
    assert torch.equal(mdp.excess_xy_drift(env), torch.tensor([True]))
