"""Headless smoke checks for Taluy normalized body-wrench control wiring."""

from __future__ import annotations

from dataclasses import dataclass
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

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    BodyWrenchAction,
    make_taluy_base_env_cfg,
)


SURGE_ACTION = 0.20
SWAY_ACTION = 0.20
HEAVE_ACTION = 0.20
ROLL_ACTION = 0.25
PITCH_ACTION = 0.30
YAW_ACTION = 0.125
ZERO_TRIM_LIN_MAX_M_S = 0.06
ZERO_TRIM_ANG_MAX_RAD_S = 0.06
MIN_LINEAR_RESPONSE_M_S = 0.025
MIN_ANGULAR_RESPONSE_RAD_S = 0.025


@dataclass(frozen=True)
class AxisCase:
    name: str
    action: tuple[float, float, float, float, float, float]
    response_kind: str
    response_index: int
    min_response: float


AXIS_CASES = (
    AxisCase(
        name="surge",
        action=(SURGE_ACTION, 0.0, 0.0, 0.0, 0.0, 0.0),
        response_kind="linear",
        response_index=0,
        min_response=MIN_LINEAR_RESPONSE_M_S,
    ),
    AxisCase(
        name="sway",
        action=(0.0, SWAY_ACTION, 0.0, 0.0, 0.0, 0.0),
        response_kind="linear",
        response_index=1,
        min_response=MIN_LINEAR_RESPONSE_M_S,
    ),
    AxisCase(
        name="heave",
        action=(0.0, 0.0, HEAVE_ACTION, 0.0, 0.0, 0.0),
        response_kind="linear",
        response_index=2,
        min_response=MIN_LINEAR_RESPONSE_M_S,
    ),
    AxisCase(
        name="roll",
        action=(0.0, 0.0, 0.0, ROLL_ACTION, 0.0, 0.0),
        response_kind="angular",
        response_index=0,
        min_response=MIN_ANGULAR_RESPONSE_RAD_S,
    ),
    AxisCase(
        name="pitch",
        action=(0.0, 0.0, 0.0, 0.0, PITCH_ACTION, 0.0),
        response_kind="angular",
        response_index=1,
        min_response=MIN_ANGULAR_RESPONSE_RAD_S,
    ),
    AxisCase(
        name="yaw",
        action=(0.0, 0.0, 0.0, 0.0, 0.0, YAW_ACTION),
        response_kind="angular",
        response_index=2,
        min_response=MIN_ANGULAR_RESPONSE_RAD_S,
    ),
)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _make_env() -> ManagerBasedRlEnv:
    cfg = make_taluy_base_env_cfg(
        action_space="body_wrench",
    )
    return ManagerBasedRlEnv(cfg=cfg, device=_device())


def _assert_finite_env_state(env: ManagerBasedRlEnv) -> None:
    if not torch.isfinite(env.sim.data.qpos).all():
        raise AssertionError("Non-finite qpos detected")
    if not torch.isfinite(env.sim.data.qvel).all():
        raise AssertionError("Non-finite qvel detected")


def _tail_mean(samples: torch.Tensor, fraction: float = 0.35) -> torch.Tensor:
    if samples.shape[0] == 0:
        raise AssertionError("Expected non-empty phase samples")
    count = max(int(round(samples.shape[0] * fraction)), 1)
    return samples[-count:].mean(dim=0)


def _leading_mean(samples: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    if samples.shape[0] == 0:
        raise AssertionError("Expected non-empty phase samples")
    count = max(int(round(samples.shape[0] * fraction)), 1)
    return samples[:count].mean(dim=0)


def _run_phase(
    env: ManagerBasedRlEnv,
    *,
    action_6: torch.Tensor,
    duration_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if action_6.shape != (6,):
        raise ValueError(f"Expected action shape (6,), got {tuple(action_6.shape)}")

    action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim),
        device=env.device,
    )
    action[:, :6] = action_6.view(1, 6)

    num_steps = max(int(duration_s / env.step_dt), 1)
    lin_hist: list[torch.Tensor] = []
    ang_hist: list[torch.Tensor] = []

    for _ in range(num_steps):
        env.step(action)
        _assert_finite_env_state(env)

        robot = env.scene["robot"]
        lin_hist.append(robot.data.root_link_lin_vel_b[0].detach().cpu())
        ang_hist.append(robot.data.root_link_ang_vel_b[0].detach().cpu())

    return torch.stack(lin_hist), torch.stack(ang_hist)


def _check_action_term_interface(env: ManagerBasedRlEnv) -> None:
    if env.action_manager.total_action_dim != 6:
        raise AssertionError(
            "Body-wrench Taluy env should expose exactly 6 policy actions. "
            f"Observed {env.action_manager.total_action_dim}."
        )

    action_term = env.action_manager.get_term("body_wrench")
    if not isinstance(action_term, BodyWrenchAction):
        raise AssertionError("Expected 'body_wrench' term to be BodyWrenchAction.")

    allocation = action_term.allocation_matrix_b.detach().cpu()
    if allocation.shape != (6, 8):
        raise AssertionError(
            "Taluy body-wrench action should expose a 6x8 allocation matrix. "
            f"Observed {tuple(allocation.shape)}."
        )
    if int(torch.linalg.matrix_rank(allocation).item()) != 6:
        raise AssertionError("Taluy allocation matrix is not full rank for 6D control.")


def _check_zero_wrench_trim(env: ManagerBasedRlEnv) -> None:
    env.reset()
    zero = torch.zeros(6, device=env.device)
    lin_hist, ang_hist = _run_phase(env, action_6=zero, duration_s=1.0)

    max_lin_speed = float(torch.linalg.norm(lin_hist, dim=1).max().item())
    max_ang_speed = float(torch.linalg.norm(ang_hist, dim=1).max().item())
    if max_lin_speed > ZERO_TRIM_LIN_MAX_M_S:
        raise AssertionError(
            "Zero-wrench linear drift too high. "
            f"Observed max linear speed {max_lin_speed:.4f} m/s"
        )
    if max_ang_speed > ZERO_TRIM_ANG_MAX_RAD_S:
        raise AssertionError(
            "Zero-wrench angular drift too high. "
            f"Observed max angular speed {max_ang_speed:.4f} rad/s"
        )


def _check_axis_case(
    env: ManagerBasedRlEnv, case: AxisCase
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    env.reset()
    zero = torch.zeros(6, device=env.device)
    action = torch.tensor(case.action, dtype=torch.float, device=env.device)

    _run_phase(env, action_6=zero, duration_s=0.2)
    lin_hist, ang_hist = _run_phase(env, action_6=action, duration_s=1.5)

    lin_lead = _leading_mean(lin_hist)
    ang_lead = _leading_mean(ang_hist)
    lin_tail = _tail_mean(lin_hist)
    ang_tail = _tail_mean(ang_hist)

    response = lin_lead if case.response_kind == "linear" else ang_lead
    observed = float(response[case.response_index])
    if observed <= case.min_response:
        units = "m/s" if case.response_kind == "linear" else "rad/s"
        raise AssertionError(
            f"{case.name} action did not create the expected positive "
            f"{case.response_kind} response on axis {case.response_index}. "
            f"Observed {observed:.4f} {units}; "
            f"linear_lead={lin_lead.tolist()} angular_lead={ang_lead.tolist()} "
            f"linear_tail={lin_tail.tolist()} angular_tail={ang_tail.tolist()}"
        )

    return lin_lead, ang_lead, lin_tail, ang_tail


def main() -> None:
    env = _make_env()
    results: list[
        tuple[str, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ] = []

    try:
        _check_action_term_interface(env)
        _check_zero_wrench_trim(env)
        for case in AXIS_CASES:
            lin_lead, ang_lead, lin_tail, ang_tail = _check_axis_case(env, case)
            results.append((case.name, lin_lead, ang_lead, lin_tail, ang_tail))
    finally:
        env.close()

    print("Taluy MJLab body-wrench smoke checks passed.")
    for name, lin_lead, ang_lead, lin_tail, ang_tail in results:
        print(
            f"{name}: lead_lin={lin_lead.tolist()} lead_ang={ang_lead.tolist()} "
            f"tail_lin={lin_tail.tolist()} tail_ang={ang_tail.tolist()}"
        )


if __name__ == "__main__":
    main()
