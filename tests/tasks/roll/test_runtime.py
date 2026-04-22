from __future__ import annotations

import math

import torch

from mjlab.utils.lab_api.math import quat_from_euler_xyz

from auvrl.tasks.roll.runtime import (
    get_roll_task_state,
    normalized_phi_total,
    quat_wxyz_to_roll_pitch_yaw,
    reset_roll_task_state,
    settle_condition_mask,
    update_phi_total,
    update_success_tracking,
)


def test_quat_wxyz_to_roll_pitch_yaw_matches_input_angles() -> None:
    roll = torch.tensor([0.60], dtype=torch.float)
    pitch = torch.tensor([-0.25], dtype=torch.float)
    yaw = torch.tensor([1.10], dtype=torch.float)
    quat = quat_from_euler_xyz(roll, pitch, yaw)

    roll_out, pitch_out, yaw_out = quat_wxyz_to_roll_pitch_yaw(quat)

    assert torch.allclose(roll_out, roll)
    assert torch.allclose(pitch_out, pitch)
    assert torch.allclose(yaw_out, yaw)


def test_update_phi_total_unwraps_across_pi_boundary() -> None:
    phi_total = torch.tensor([0.0], dtype=torch.float)
    prev_roll = torch.tensor([math.radians(170.0)], dtype=torch.float)
    current_roll = torch.tensor([math.radians(-170.0)], dtype=torch.float)

    next_phi_total, next_prev_roll, delta_roll = update_phi_total(
        phi_total,
        prev_roll,
        current_roll,
    )

    assert torch.allclose(delta_roll, torch.tensor([math.radians(20.0)]))
    assert torch.allclose(next_phi_total, torch.tensor([math.radians(20.0)]))
    assert torch.allclose(next_prev_roll, current_roll)


def test_normalized_phi_total_applies_direction_and_clip() -> None:
    phi_total = torch.tensor([2.0 * math.pi, -2.0 * math.pi], dtype=torch.float)

    normalized = normalized_phi_total(
        phi_total,
        target_roll_rad=2.0 * math.pi,
        roll_direction=1,
    )
    reversed_normalized = normalized_phi_total(
        phi_total,
        target_roll_rad=2.0 * math.pi,
        roll_direction=-1,
    )

    assert torch.allclose(normalized, torch.tensor([1.0, -1.0]))
    assert torch.allclose(reversed_normalized, torch.tensor([-1.0, 1.0]))

    clipped = normalized_phi_total(
        torch.tensor([10.0 * math.pi]),
        target_roll_rad=2.0 * math.pi,
        roll_direction=1,
    )
    assert torch.allclose(clipped, torch.tensor([2.0]))


def test_settle_condition_mask_checks_all_constraints() -> None:
    mask = settle_condition_mask(
        pitch_rad=torch.tensor([0.05, 0.30]),
        yaw_error_rad=torch.tensor([0.10, 0.10]),
        ang_vel_b_rad_s=torch.tensor([[0.05, 0.10, 0.15], [0.30, 0.0, 0.0]]),
        depth_error_m=torch.tensor([0.05, 0.05]),
        pitch_limit_rad=math.radians(10.0),
        yaw_limit_rad=math.radians(15.0),
        ang_vel_limit_rad_s=0.25,
        depth_error_limit_m=0.15,
    )

    assert torch.equal(mask, torch.tensor([True, False]))


def test_update_success_tracking_requires_continuous_settle_window() -> None:
    phi_total = torch.tensor([4.0 * math.pi], dtype=torch.float)
    target_reached = torch.tensor([False])
    settle_counter = torch.tensor([0], dtype=torch.long)

    target_reached, settle_counter, success = update_success_tracking(
        phi_total_rad=phi_total,
        target_reached=target_reached,
        settle_counter_steps=settle_counter,
        settle_mask=torch.tensor([True]),
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
        settle_steps=2,
    )
    assert torch.equal(target_reached, torch.tensor([True]))
    assert torch.equal(settle_counter, torch.tensor([1]))
    assert torch.equal(success, torch.tensor([False]))

    target_reached, settle_counter, success = update_success_tracking(
        phi_total_rad=phi_total,
        target_reached=target_reached,
        settle_counter_steps=settle_counter,
        settle_mask=torch.tensor([False]),
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
        settle_steps=2,
    )
    assert torch.equal(target_reached, torch.tensor([True]))
    assert torch.equal(settle_counter, torch.tensor([0]))
    assert torch.equal(success, torch.tensor([False]))

    target_reached, settle_counter, success = update_success_tracking(
        phi_total_rad=phi_total,
        target_reached=target_reached,
        settle_counter_steps=settle_counter,
        settle_mask=torch.tensor([True]),
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
        settle_steps=2,
    )
    target_reached, settle_counter, success = update_success_tracking(
        phi_total_rad=phi_total,
        target_reached=target_reached,
        settle_counter_steps=settle_counter,
        settle_mask=torch.tensor([True]),
        target_roll_rad=4.0 * math.pi,
        roll_direction=1,
        settle_steps=2,
    )

    assert torch.equal(success, torch.tensor([True]))


def test_reset_roll_task_state_re_latches_even_if_episode_counter_is_nonzero() -> None:
    class DummyEnv:
        def __init__(self) -> None:
            self.device = "cpu"
            self.num_envs = 1
            self.episode_length_buf = torch.tensor([37], dtype=torch.long)
            self.scene = {
                "robot": type(
                    "Robot",
                    (),
                    {
                        "data": type(
                            "Data",
                            (),
                            {
                                "root_link_pos_w": torch.tensor([[1.0, 2.0, -0.3]]),
                                "root_link_quat_w": quat_from_euler_xyz(
                                    torch.tensor([0.4]),
                                    torch.tensor([0.0]),
                                    torch.tensor([0.8]),
                                ),
                                "root_link_ang_vel_b": torch.zeros(1, 3),
                            },
                        )()
                    },
                )()
            }

    env = DummyEnv()
    env.episode_length_buf[:] = 0
    state = get_roll_task_state(env)
    assert torch.equal(state.last_update_step, torch.tensor([0]))

    env.episode_length_buf[:] = 37
    env.scene["robot"].data.root_link_pos_w = torch.tensor([[3.0, -1.0, -0.8]])
    env.scene["robot"].data.root_link_quat_w = quat_from_euler_xyz(
        torch.tensor([-0.2]),
        torch.tensor([0.0]),
        torch.tensor([-0.6]),
    )
    state.phi_total_rad[:] = 5.0
    state.target_reached[:] = True
    state.settle_counter_steps[:] = 9

    reset_roll_task_state(env, env_ids=None)

    assert torch.equal(state.last_update_step, torch.tensor([0]))
    assert torch.allclose(state.phi_total_rad, torch.zeros(1))
    assert torch.equal(state.target_reached, torch.tensor([False]))
    assert torch.equal(state.settle_counter_steps, torch.tensor([0]))
    assert torch.allclose(state.z_ref_m, torch.tensor([-0.8]))
    assert torch.allclose(state.xy_ref_w, torch.tensor([[3.0, -1.0]]))
