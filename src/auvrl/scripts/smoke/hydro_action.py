"""Smoke checks for the project-local MJLab underwater hydro action term."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.utils.lab_api.math import (  # type: ignore[import-not-found]
        quat_apply,
        quat_apply_inverse,
        quat_from_angle_axis,
        quat_mul,
        skew_symmetric_matrix,
    )

    from auvrl import (  # type: ignore[import-not-found]
        HydroConfig,
        UnderwaterHydroActionCfg,
    )
    from auvrl.sim.hydrodynamics import (  # type: ignore[import-not-found]
        AUVBodyState,
        HydrodynamicsModel,
        added_mass_coriolis_wrench,
        shift_wrench_com_to_origin,
        shift_wrench_origin_to_com,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import auvrl/mjlab modules. Ensure deps are installed and"
        " package is available in this workspace."
    ) from exc


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _diag6(values: tuple[float, float, float, float, float, float]):
    return (
        (values[0], 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, values[1], 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, values[2], 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, values[3], 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, values[4], 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, values[5]),
    )


def _make_mock_env(
    device: str,
    num_envs: int = 1,
    physics_dt: float = 0.1,
    mass_kg: float = 1.0,
    center_of_gravity_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    body_quat = torch.zeros((num_envs, 1, 4), device=device)
    body_quat[..., 0] = 1.0

    body_mass = torch.full((num_envs, 1), mass_kg, device=device)
    body_ipos = (
        torch.tensor(center_of_gravity_b, dtype=torch.float32, device=device)
        .view(1, 1, 3)
        .expand(num_envs, -1, -1)
        .clone()
    )

    entity = Mock()
    entity.find_bodies.return_value = ([0], ["base_link"])
    entity.data = SimpleNamespace(
        body_link_quat_w=body_quat,
        body_link_lin_vel_w=torch.zeros((num_envs, 1, 3), device=device),
        body_link_ang_vel_w=torch.zeros((num_envs, 1, 3), device=device),
        indexing=SimpleNamespace(body_ids=torch.tensor([0], device=device)),
        model=SimpleNamespace(body_mass=body_mass, body_ipos=body_ipos),
    )
    entity.write_external_wrench_to_sim = Mock()

    env = SimpleNamespace(
        num_envs=num_envs,
        device=device,
        physics_dt=physics_dt,
        scene={"robot": entity},
    )
    return env, entity


def _wrench_from_last_call(entity: Mock) -> torch.Tensor:
    call_args = entity.write_external_wrench_to_sim.call_args
    forces = call_args.args[0].squeeze(1)
    torques = call_args.args[1].squeeze(1)
    return torch.cat((forces, torques), dim=1)


def _check_action_dim(device: str) -> None:
    env, _ = _make_mock_env(device)
    cfg = UnderwaterHydroActionCfg(entity_name="robot", body_name="base_link")
    term = cfg.build(cast(Any, env))
    assert term.action_dim == 0
    assert term.raw_action.shape == (1, 0)


def _check_full_hydro_matches_mujoco_model(device: str) -> None:
    mass_kg = 11.0
    center_of_gravity_b = (-0.02, 0.01, -0.06)
    env, entity = _make_mock_env(
        device,
        physics_dt=0.05,
        mass_kg=mass_kg,
        center_of_gravity_b=center_of_gravity_b,
    )

    quat_wb = torch.tensor(
        [[0.94832367, 0.12963414, -0.19445121, 0.19445121]],
        dtype=torch.float,
        device=device,
    )
    quat_wb = quat_wb / torch.linalg.norm(quat_wb, dim=1, keepdim=True)
    entity.data.body_link_quat_w[:, 0] = quat_wb

    initial_lin_b = torch.tensor([[0.4, -0.2, 0.1]], dtype=torch.float, device=device)
    initial_ang_b = torch.tensor(
        [[0.05, -0.04, 0.03]], dtype=torch.float, device=device
    )
    next_lin_b = torch.tensor([[0.7, -0.1, 0.25]], dtype=torch.float, device=device)
    next_ang_b = torch.tensor([[0.15, -0.02, 0.08]], dtype=torch.float, device=device)

    avg_ang_b = 0.5 * (initial_ang_b + next_ang_b)
    avg_ang_norm = torch.linalg.norm(avg_ang_b, dim=1)
    avg_ang_axis = avg_ang_b / avg_ang_norm.unsqueeze(1)
    delta_quat = quat_from_angle_axis(avg_ang_norm * env.physics_dt, avg_ang_axis)
    next_quat_wb = quat_mul(delta_quat, quat_wb)
    next_quat_wb = next_quat_wb / torch.linalg.norm(next_quat_wb, dim=1, keepdim=True)

    entity.data.body_link_lin_vel_w[:, 0] = quat_apply(quat_wb, initial_lin_b)
    entity.data.body_link_ang_vel_w[:, 0] = quat_apply(quat_wb, initial_ang_b)

    added_mass_6x6 = (
        (1.8, 0.1, 0.0, 0.0, 0.2, 0.0),
        (0.1, 2.1, -0.1, -0.2, 0.0, 0.0),
        (0.0, -0.1, 2.7, 0.0, 0.1, -0.1),
        (0.0, -0.2, 0.0, 0.4, 0.0, 0.1),
        (0.2, 0.0, 0.1, 0.0, 0.6, -0.1),
        (0.0, 0.0, -0.1, 0.1, -0.1, 0.8),
    )

    cfg = UnderwaterHydroActionCfg(
        entity_name="robot",
        body_name="base_link",
        linear_damping_matrix_6x6=(
            (10.0, 0.5, 0.0, 0.0, 0.0, 0.0),
            (0.0, 11.0, -0.3, 0.0, 0.0, 0.0),
            (0.0, 0.0, 12.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 2.0, 0.1, 0.0),
            (0.0, 0.0, 0.0, 0.0, 3.0, 0.2),
            (0.0, 0.0, 0.0, 0.0, 0.0, 4.0),
        ),
        quadratic_damping_matrix_6x6=(
            (1.5, 0.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.2, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.3, 0.0),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.4),
        ),
        current_velocity_w=(0.3, -0.2, 0.1),
        fluid_density_kg_m3=1000.0,
        gravity_m_s2=9.81,
        buoyancy_n=117.72,
        center_of_buoyancy_b_m=(0.04, -0.03, 0.05),
        added_mass_6x6=added_mass_6x6,
        include_damping=True,
        include_restoring=True,
        include_added_mass=True,
        include_added_coriolis=True,
    )
    term = cfg.build(cast(Any, env))

    term.apply_actions()

    entity.data.body_link_quat_w[:, 0] = next_quat_wb
    entity.data.body_link_lin_vel_w[:, 0] = quat_apply(next_quat_wb, next_lin_b)
    entity.data.body_link_ang_vel_w[:, 0] = quat_apply(next_quat_wb, next_ang_b)
    term.apply_actions()

    actual_wrench_w = _wrench_from_last_call(entity)

    dt_s = float(env.physics_dt)
    lin_acc_b = (next_lin_b - initial_lin_b) / dt_s
    ang_acc_b = (next_ang_b - initial_ang_b) / dt_s
    current_velocity_w = torch.tensor(
        [[0.3, -0.2, 0.1]], dtype=torch.float, device=device
    )
    initial_relative_twist_b = torch.cat(
        (
            initial_lin_b - quat_apply_inverse(quat_wb, current_velocity_w),
            initial_ang_b,
        ),
        dim=1,
    )
    next_relative_twist_b = torch.cat(
        (
            next_lin_b - quat_apply_inverse(next_quat_wb, current_velocity_w),
            next_ang_b,
        ),
        dim=1,
    )
    relative_twist_dot_b = (next_relative_twist_b - initial_relative_twist_b) / dt_s

    hydro_model = HydrodynamicsModel(
        HydroConfig(
            weight_n=mass_kg * 9.81,
            center_of_gravity_b_m=center_of_gravity_b,
            center_of_buoyancy_b_m=(0.04, -0.03, 0.05),
            current_world_m_s=(0.3, -0.2, 0.1),
            fluid_density_kg_m3=1000.0,
            gravity_m_s2=9.81,
            buoyancy_n=117.72,
            linear_damping_matrix_6x6=cfg.linear_damping_matrix_6x6,
            quadratic_damping_matrix_6x6=cfg.quadratic_damping_matrix_6x6,
            added_mass_6x6=added_mass_6x6,
            include_damping=True,
            include_restoring=True,
            include_added_mass=True,
            include_added_coriolis=True,
        ),
        device=device,
    )
    expected_wrench_b, _ = hydro_model.compute_wrench(
        AUVBodyState(
            quat_wxyz=next_quat_wb,
            relative_twist_dot_body=relative_twist_dot_b,
            lin_vel_body=next_lin_b,
            ang_vel_body=next_ang_b,
            lin_acc_body=lin_acc_b,
            ang_acc_body=ang_acc_b,
        )
    )
    expected_wrench_com_b = shift_wrench_origin_to_com(
        expected_wrench_b,
        torch.tensor([center_of_gravity_b], dtype=torch.float32, device=device),
    )
    expected_wrench_w = torch.cat(
        (
            quat_apply(next_quat_wb, expected_wrench_com_b[:, :3]),
            quat_apply(next_quat_wb, expected_wrench_com_b[:, 3:]),
        ),
        dim=1,
    )

    torch.testing.assert_close(actual_wrench_w, expected_wrench_w, atol=1e-5, rtol=1e-5)


def _check_body_frame_current(device: str) -> None:
    env, entity = _make_mock_env(device)

    quat_wb = torch.tensor(
        [[0.70710677, 0.0, 0.0, 0.70710677]],
        dtype=torch.float,
        device=device,
    )
    entity.data.body_link_quat_w[:, 0] = quat_wb

    cfg = UnderwaterHydroActionCfg(
        entity_name="robot",
        body_name="base_link",
        linear_damping_matrix_6x6=_diag6((5.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        quadratic_damping_matrix_6x6=_diag6((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        current_velocity_b=(1.0, 0.0, 0.0),
        include_restoring=False,
    )
    term = cfg.build(cast(Any, env))
    term.apply_actions()

    wrench_w = _wrench_from_last_call(entity)
    expected_force_w = quat_apply(
        quat_wb,
        torch.tensor([[5.0, 0.0, 0.0]], dtype=torch.float, device=device),
    )

    torch.testing.assert_close(
        wrench_w[:, :3],
        expected_force_w,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        term.current_velocity_b,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float, device=device),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        term.current_velocity_w,
        quat_apply(
            quat_wb,
            torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float, device=device),
        ),
        atol=1e-6,
        rtol=1e-6,
    )


def _check_wrench_shift_roundtrip(device: str) -> None:
    wrench_origin = torch.tensor(
        [[4.0, -2.0, 1.5, 0.6, -0.4, 0.2]],
        dtype=torch.float32,
        device=device,
    )
    cog = torch.tensor([[0.2, -0.1, 0.05]], dtype=torch.float32, device=device)

    wrench_com = shift_wrench_origin_to_com(wrench_origin, cog)
    roundtrip = shift_wrench_com_to_origin(wrench_com, cog)

    torch.testing.assert_close(roundtrip, wrench_origin, atol=1e-6, rtol=1e-6)


def _check_added_mass_coriolis_matches_matrix_form(device: str) -> None:
    added_mass = torch.tensor(
        (
            (1.8, 0.1, 0.0, 0.0, 0.2, 0.0),
            (0.1, 2.1, -0.1, -0.2, 0.0, 0.0),
            (0.0, -0.1, 2.7, 0.0, 0.1, -0.1),
            (0.0, -0.2, 0.0, 0.4, 0.0, 0.1),
            (0.2, 0.0, 0.1, 0.0, 0.6, -0.1),
            (0.0, 0.0, -0.1, 0.1, -0.1, 0.8),
        ),
        dtype=torch.float32,
        device=device,
    )
    twist = torch.tensor(
        [[0.7, -0.1, 0.25, 0.15, -0.02, 0.08]],
        dtype=torch.float32,
        device=device,
    )

    nu1 = twist[0, :3]
    nu2 = twist[0, 3:]
    a = added_mass[:3, :3] @ nu1 + added_mass[:3, 3:] @ nu2
    b = added_mass[3:, :3] @ nu1 + added_mass[3:, 3:] @ nu2

    coriolis = torch.zeros((6, 6), dtype=torch.float32, device=device)
    coriolis[:3, 3:] = -skew_symmetric_matrix(a)[0]
    coriolis[3:, :3] = -skew_symmetric_matrix(a)[0]
    coriolis[3:, 3:] = -skew_symmetric_matrix(b)[0]

    expected = -(coriolis @ twist[0])
    actual = added_mass_coriolis_wrench(added_mass, twist)[0]

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)


def _check_reset_zeroes_wrench(device: str) -> None:
    env, entity = _make_mock_env(device, num_envs=2)

    cfg = UnderwaterHydroActionCfg(entity_name="robot", body_name="base_link")
    term = cfg.build(cast(Any, env))

    env_ids = torch.tensor([1], device=device)
    term.reset(env_ids=env_ids)

    call_args = entity.write_external_wrench_to_sim.call_args
    forces = call_args.args[0]
    torques = call_args.args[1]

    torch.testing.assert_close(forces, torch.zeros((1, 1, 3), device=device))
    torch.testing.assert_close(torques, torch.zeros((1, 1, 3), device=device))
    assert torch.equal(call_args.kwargs["env_ids"], env_ids)
    assert call_args.kwargs["body_ids"] == [0]


def main() -> None:
    device = _device()
    _check_action_dim(device)
    _check_full_hydro_matches_mujoco_model(device)
    _check_body_frame_current(device)
    _check_wrench_shift_roundtrip(device)
    _check_added_mass_coriolis_matches_matrix_form(device)
    _check_reset_zeroes_wrench(device)
    print("MJLab hydro action smoke checks passed.")


if __name__ == "__main__":
    main()
