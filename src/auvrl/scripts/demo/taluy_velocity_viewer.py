"""Interactive Taluy velocity-task viewer demo.

This launches the first Taluy 6D velocity-tracking environment with a simple
hand-tuned feedback policy so the Viser command GUI is immediately useful.

Recommended flow in Viser:
1. Open the URL printed by Viser.
2. In `Commands -> Body Velocity`, enable the GUI term and move the 6 sliders.
3. In `Scene`, enable the `Body_velocity` debug visualization to inspect the
   stacked linear-velocity arrows and per-thruster force arrows.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import inspect
import os
from pathlib import Path
from typing import Any, cast

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.viewer import (  # type: ignore[import-not-found]
        NativeMujocoViewer,
        ViserPlayViewer,
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Could not import mjlab RL/viewer dependencies. Ensure mjlab is available."
    ) from exc

from auvrl import (  # noqa: E402  # type: ignore[import-not-found]
    UniformBodyVelocityCommand,
    make_taluy_velocity_env_cfg,
)
from auvrl.actuator.body_wrench_action import (  # noqa: E402  # type: ignore[import-not-found]
    BodyWrenchAction,
)
from auvrl.config.auv_cfg import (  # noqa: E402  # type: ignore[import-not-found]
    TALUY_CFG_PATH,
    load_auv_cfg,
)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _build_viser_server(host: str, port: int):
    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    try:
        return viser.ViserServer(host=host, port=port, label="taluy-velocity-demo")
    except TypeError:
        print("Warning: this viser version does not support host/port arguments.")
        return viser.ViserServer(label="taluy-velocity-demo")


def _run_viser_viewer(
    env: RslRlVecEnvWrapper,
    policy: Callable[[object], torch.Tensor],
    host: str,
    port: int,
) -> None:
    params = inspect.signature(ViserPlayViewer.__init__).parameters

    if "viser_server" in params:
        server = _build_viser_server(host, port)
        ViserPlayViewer(env, cast(Any, policy), viser_server=server).run()  # type: ignore[call-arg]
        return

    if "server" in params:
        server = _build_viser_server(host, port)
        ViserPlayViewer(env, cast(Any, policy), server=server).run()  # type: ignore[call-arg]
        return

    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    print(
        "Using legacy MJLab viewer API. "
        "Injecting --viser-host/--viser-port via temporary Viser patch."
    )
    original_server_cls = viser.ViserServer

    def _patched_server(*args, **kwargs):
        kwargs.setdefault("host", host)
        kwargs.setdefault("port", port)
        return original_server_cls(*args, **kwargs)

    viser.ViserServer = _patched_server
    try:
        ViserPlayViewer(env, cast(Any, policy)).run()
    finally:
        viser.ViserServer = original_server_cls


class ZeroWrenchPolicy:
    def __init__(self, num_envs: int, device: str):
        self._action = torch.zeros((num_envs, 6), device=device, dtype=torch.float)

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        return self._action


class SimpleBodyVelocityTrackingPolicy:
    def __init__(
        self,
        env: RslRlVecEnvWrapper,
        *,
        linear_gains: tuple[float, float, float] = (140.0, 140.0, 180.0),
        angular_gains: tuple[float, float, float] = (55.0, 55.0, 70.0),
    ):
        self._base_env = env.unwrapped
        self._linear_gains = torch.tensor(
            linear_gains,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 3)
        self._angular_gains = torch.tensor(
            angular_gains,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 3)
        taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)
        self._wrench_limits = torch.tensor(
            taluy_cfg.body_wrench_limit,
            device=self._base_env.device,
            dtype=torch.float,
        ).view(1, 6)

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        robot = self._base_env.scene["robot"]
        command = self._base_env.command_manager.get_command("body_velocity")
        assert command is not None
        lin_error = command[:, :3] - robot.data.root_link_lin_vel_b
        ang_error = command[:, 3:] - robot.data.root_link_ang_vel_b
        wrench = torch.cat(
            (lin_error * self._linear_gains, ang_error * self._angular_gains),
            dim=1,
        )
        wrench = torch.clamp(wrench, min=-self._wrench_limits, max=self._wrench_limits)
        return wrench / self._wrench_limits


class InspectablePolicy:
    def __init__(
        self,
        base_policy: Callable[[object], torch.Tensor],
        base_env: ManagerBasedRlEnv,
        print_period_steps: int,
    ) -> None:
        self._base_policy = base_policy
        self._base_env = base_env
        self._print_period_steps = max(int(print_period_steps), 1)
        self._step = 0

    def __call__(self, obs: object) -> torch.Tensor:
        actions = self._base_policy(obs)

        if self._step % self._print_period_steps == 0:
            robot = self._base_env.scene["robot"]
            command = self._base_env.command_manager.get_command("body_velocity")
            assert command is not None
            wrench_term = self._base_env.action_manager.get_term("body_wrench")
            assert isinstance(wrench_term, BodyWrenchAction)
            policy_wrench = wrench_term.action_to_wrench(actions)

            lin_vel = robot.data.root_link_lin_vel_b[0].detach().cpu().tolist()
            ang_vel = robot.data.root_link_ang_vel_b[0].detach().cpu().tolist()
            desired_wrench = wrench_term.desired_wrench_b[0].detach().cpu().tolist()
            max_thruster = float(wrench_term.thruster_targets[0].abs().max().item())
            saturation_fraction = float(wrench_term.step_saturation_fraction[0].item())

            print(
                "[live] "
                f"command_b={command[0].detach().cpu().tolist()} "
                f"lin_vel_b={lin_vel} ang_vel_b={ang_vel} "
                f"policy_action={actions[0].detach().cpu().tolist()} "
                f"policy_wrench_b={policy_wrench[0].detach().cpu().tolist()} "
                f"last_applied_wrench_b={desired_wrench} "
                f"last_thruster_max_n={max_thruster:.3f} "
                f"last_sat_frac={saturation_fraction:.3f}"
            )

        self._step += 1
        return actions


def _set_fixed_command(base_env: ManagerBasedRlEnv, values: tuple[float, ...]) -> None:
    if len(values) != 6:
        raise ValueError(f"Expected 6 command values, got {len(values)}.")

    term = base_env.command_manager.get_term("body_velocity")
    if not isinstance(term, UniformBodyVelocityCommand):
        raise RuntimeError("Taluy body_velocity command term is not available.")

    command = torch.tensor(values, device=base_env.device, dtype=torch.float).view(1, 6)
    term.vel_command_b[:] = command.expand(base_env.num_envs, -1)
    term.is_zero_env[:] = False
    term.time_left[:] = 1.0e9


def _run_dry_steps(
    env: RslRlVecEnvWrapper,
    policy: Callable[[object], torch.Tensor],
    *,
    num_steps: int,
    fixed_command: tuple[float, ...],
) -> None:
    base_env = env.unwrapped
    env.reset()
    _set_fixed_command(base_env, fixed_command)

    reward = torch.zeros(env.num_envs, device=base_env.device)
    for _ in range(num_steps):
        obs = env.get_observations()
        action = policy(obs)
        _, reward, _, _ = env.step(action)

    robot = base_env.scene["robot"]
    lin_vel = robot.data.root_link_lin_vel_b.detach().cpu()[0].tolist()
    ang_vel = robot.data.root_link_ang_vel_b.detach().cpu()[0].tolist()
    command = base_env.command_manager.get_command("body_velocity")
    assert command is not None
    print("Dry-run complete.")
    print(f"  command_b={command.detach().cpu()[0].tolist()}")
    print(f"  lin_vel_b={lin_vel}")
    print(f"  ang_vel_b={ang_vel}")
    print(f"  reward={reward.detach().cpu().tolist()}")


def _print_debug_legend() -> None:
    print(
        "Debug legend: purple = commanded linear velocity, blue = measured "
        "linear velocity; both share one anchor above the robot center."
    )
    print(
        "             orange/pink thruster arrows = commanded thrust along each "
        "thruster's local -Z force axis; sub-1 N commands are hidden."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="viser",
        help="Viewer backend. Use `viser` for the command GUI.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device.",
    )
    parser.add_argument(
        "--policy",
        choices=("simple", "zero"),
        default="simple",
        help="Demo policy. `simple` is a hand-tuned body-velocity feedback baseline.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel env count.",
    )
    parser.add_argument(
        "--viser-host",
        default="0.0.0.0",
        help="Viser host address.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=9000,
        help="Viser port.",
    )
    parser.add_argument(
        "--dry-run-steps",
        type=int,
        default=0,
        help="Run a short headless rollout instead of opening a viewer.",
    )
    parser.add_argument(
        "--dry-command",
        type=float,
        nargs=6,
        default=(0.30, 0.0, 0.0, 0.0, 0.0, 0.0),
        metavar=("VX", "VY", "VZ", "WX", "WY", "WZ"),
        help="Fixed body velocity command used during --dry-run-steps.",
    )
    parser.add_argument(
        "--print-period-s",
        type=float,
        default=1.0,
        help="Live print period. Set <=0 to disable console prints.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    viewer = _resolve_viewer(args.viewer)

    base_env = ManagerBasedRlEnv(
        cfg=make_taluy_velocity_env_cfg(
            num_envs=args.num_envs,
            episode_length_s=300.0,
            command_resampling_time_s=(30.0, 30.0),
            command_rel_zero_envs=1.0,
            command_init_velocity_prob=0.0,
            command_lin_vel_x_range_m_s=(-0.6, 0.6),
            command_lin_vel_y_range_m_s=(-0.6, 0.6),
            command_lin_vel_z_range_m_s=(-0.4, 0.4),
            command_ang_vel_x_range_rad_s=(-1.0, 1.0),
            command_ang_vel_y_range_rad_s=(-1.0, 1.0),
            command_ang_vel_z_range_rad_s=(-1.2, 1.2),
        ),
        device=device,
    )
    env = RslRlVecEnvWrapper(base_env, clip_actions=1.0)

    if args.policy == "zero":
        policy: Callable[[object], torch.Tensor] = ZeroWrenchPolicy(
            num_envs=env.num_envs,
            device=device,
        )
    else:
        policy = SimpleBodyVelocityTrackingPolicy(env)

    if args.print_period_s > 0.0:
        print_period_steps = max(int(round(args.print_period_s / base_env.step_dt)), 1)
        policy = InspectablePolicy(
            policy,
            base_env,
            print_period_steps=print_period_steps,
        )

    try:
        if args.dry_run_steps > 0:
            _run_dry_steps(
                env,
                policy,
                num_steps=args.dry_run_steps,
                fixed_command=tuple(float(v) for v in args.dry_command),
            )
            return

        print(f"Taluy velocity viewer demo | device={device} viewer={viewer}")
        print(f"Policy: {args.policy}")
        _print_debug_legend()
        if viewer == "viser":
            print(
                "Use `Commands -> Body Velocity` to set the 6D reference, and "
                "`Scene` to toggle the debug overlays."
            )
        else:
            print("Native viewer does not expose the Viser command GUI.")

        if viewer == "native":
            NativeMujocoViewer(env, policy).run()
        else:
            _run_viser_viewer(env, policy, args.viser_host, args.viser_port)
    finally:
        env.close()


if __name__ == "__main__":
    main()
