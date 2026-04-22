from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv

from auvrl import make_taluy_roll_env_cfg
from auvrl.tasks.roll.mdp.observations import (
    base_quat_wxyz,
    depth_error_from_ref,
    last_body_wrench_action,
    phi_total_norm,
    xy_error_w,
)
from auvrl.tasks.roll.runtime import get_roll_task_state


class FakeActionManager:
    def __init__(
        self,
        *,
        active_terms: list[str],
        action_term_dim: list[int],
        prev_action: torch.Tensor,
    ) -> None:
        self.active_terms = active_terms
        self.action_term_dim = action_term_dim
        self.prev_action = prev_action


class FakeEnv:
    def __init__(
        self,
        *,
        qpos: torch.Tensor,
        stale_root_pos_w: torch.Tensor,
        stale_root_quat_w: torch.Tensor,
        action_manager: FakeActionManager | None = None,
    ) -> None:
        self.device = qpos.device
        self.num_envs = qpos.shape[0]
        self.episode_length_buf = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=qpos.device,
        )
        self.scene = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    data=SimpleNamespace(qpos=qpos),
                    indexing=SimpleNamespace(free_joint_q_adr=torch.arange(1, 8)),
                    root_link_pos_w=stale_root_pos_w,
                    root_link_quat_w=stale_root_quat_w,
                )
            )
        }
        if action_manager is not None:
            self.action_manager = action_manager


def _make_fake_env(num_envs: int = 2) -> FakeEnv:
    qpos = torch.zeros(num_envs, 8, dtype=torch.float)
    seed_pos_w = torch.tensor(
        [[1.0, -2.0, -0.40], [2.0, 3.0, -0.75]],
        dtype=torch.float,
    )
    seed_quat_wxyz = torch.tensor(
        [[0.5, 0.5, 0.5, 0.5], [0.70710677, 0.70710677, 0.0, 0.0]],
        dtype=torch.float,
    )
    repeat_count = math.ceil(num_envs / seed_pos_w.shape[0])
    qpos[:, 1:4] = seed_pos_w.repeat((repeat_count, 1))[:num_envs]
    qpos[:, 4:8] = seed_quat_wxyz.repeat((repeat_count, 1))[:num_envs]
    stale_root_pos_w = torch.full((num_envs, 3), 42.0, dtype=torch.float)
    stale_root_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * num_envs)
    return FakeEnv(
        qpos=qpos,
        stale_root_pos_w=stale_root_pos_w,
        stale_root_quat_w=stale_root_quat_w,
    )


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def test_base_quat_wxyz_reads_qpos_not_stale_root_link_quat() -> None:
    env = _make_fake_env()

    quat = base_quat_wxyz(env)

    assert torch.allclose(quat, env.scene["robot"].data.data.qpos[:, 4:8])
    assert not torch.allclose(quat, env.scene["robot"].data.root_link_quat_w)


def test_depth_error_from_ref_uses_current_z_minus_reference_with_column_shape() -> None:
    env = _make_fake_env()
    state = get_roll_task_state(env)

    env.scene["robot"].data.data.qpos[:, 3] = state.z_ref_m + torch.tensor(
        [0.25, -0.50],
        dtype=torch.float,
    )

    depth_error = depth_error_from_ref(env)

    assert depth_error.shape == (2, 1)
    assert torch.allclose(depth_error, torch.tensor([[0.25], [-0.50]]))


def test_xy_error_w_uses_current_xy_minus_reference_with_xy_shape() -> None:
    env = _make_fake_env()
    state = get_roll_task_state(env)

    env.scene["robot"].data.data.qpos[:, 1:3] = state.xy_ref_w + torch.tensor(
        [[0.10, -0.20], [-0.30, 0.40]],
        dtype=torch.float,
    )

    xy_error = xy_error_w(env)

    assert xy_error.shape == (2, 2)
    assert torch.allclose(
        xy_error,
        torch.tensor([[0.10, -0.20], [-0.30, 0.40]], dtype=torch.float),
    )


def test_phi_total_norm_scales_and_clips_for_both_roll_directions() -> None:
    env = _make_fake_env(num_envs=5)
    state = get_roll_task_state(env)
    state.phi_total_rad[:] = torch.tensor(
        [-30.0, -5.0, 0.0, 5.0, 25.0],
        dtype=torch.float,
    )

    positive = phi_total_norm(env, target_roll_rad=10.0, roll_direction=1)
    negative = phi_total_norm(env, target_roll_rad=10.0, roll_direction=-1)

    assert positive.shape == (5, 1)
    assert negative.shape == (5, 1)
    assert torch.allclose(
        positive.squeeze(1),
        torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0]),
    )
    assert torch.allclose(
        negative.squeeze(1),
        torch.tensor([2.0, 0.5, 0.0, -0.5, -2.0]),
    )


def test_last_body_wrench_action_returns_named_slice_with_multiple_action_terms() -> None:
    prev_action = torch.arange(22, dtype=torch.float).reshape(2, 11)
    env = _make_fake_env()
    env.action_manager = FakeActionManager(
        active_terms=["hydro", "body_wrench", "fin_trim"],
        action_term_dim=[2, 6, 3],
        prev_action=prev_action,
    )

    body_wrench = last_body_wrench_action(env)

    assert body_wrench.shape == (2, 6)
    assert torch.equal(body_wrench, prev_action[:, 2:8])


def test_last_body_wrench_action_raises_when_named_action_term_is_missing() -> None:
    env = _make_fake_env()
    env.action_manager = FakeActionManager(
        active_terms=["hydro", "fin_trim"],
        action_term_dim=[2, 3],
        prev_action=torch.zeros(2, 5),
    )

    with pytest.raises(ValueError, match="Action term 'body_wrench' not found"):
        last_body_wrench_action(env)


def test_taluy_roll_env_observation_contract_and_previous_action_semantics() -> None:
    num_envs = 2
    cfg = make_taluy_roll_env_cfg(num_envs=num_envs)
    env = ManagerBasedRlEnv(cfg=cfg, device=_device())
    try:
        obs, _ = env.reset()
        actor = obs["actor"]
        critic = obs["critic"]

        assert actor.shape == (num_envs, 15)
        assert critic.shape == (num_envs, 20)
        assert torch.allclose(critic[:, :15], actor)
        assert torch.allclose(actor[:, 7:8], torch.zeros_like(actor[:, 7:8]))
        assert torch.allclose(actor[:, 8:9], torch.zeros_like(actor[:, 8:9]))
        assert torch.allclose(actor[:, 9:15], torch.zeros_like(actor[:, 9:15]))
        assert torch.allclose(critic[:, 18:20], torch.zeros_like(critic[:, 18:20]))

        first_action = torch.zeros(
            (num_envs, env.action_manager.total_action_dim),
            device=env.device,
        )
        first_action[:, 3] = 0.25
        second_action = torch.zeros_like(first_action)

        first_obs, *_ = env.step(first_action)
        assert torch.allclose(
            first_obs["actor"][:, 9:15],
            torch.zeros_like(first_obs["actor"][:, 9:15]),
            atol=1.0e-6,
        )

        second_obs, *_ = env.step(second_action)
        assert torch.allclose(
            second_obs["actor"][:, 9:15],
            first_action,
            atol=1.0e-6,
        )
    finally:
        env.close()
