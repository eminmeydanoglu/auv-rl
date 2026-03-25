"""Play a trained Taluy 6D velocity policy in a viewer.

This loads a PPO checkpoint produced by
`src/auvrl/scripts/train/taluy_velocity.py`, opens the environment in a viewer,
and steps it with the learned policy so you can visually inspect behavior.

Recommended flow in Viser:
1. Open the URL printed by Viser.
2. In `Commands -> Body Velocity`, enable the GUI term.
3. Move the 6 sliders and watch the policy track the reference.
4. In `Scene`, enable the `Body_velocity` debug visualization.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict
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

import yaml

ROOT = Path(__file__).resolve().parents[4]

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.os import get_checkpoint_path  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
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
    taluy_velocity_ppo_runner_cfg,
)
from auvrl.actuator.body_wrench_action import BodyWrenchAction  # noqa: E402  # type: ignore[import-not-found]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-file",
        default=None,
        help="Direct path to a checkpoint file. Overrides experiment/run lookup.",
    )
    parser.add_argument(
        "--experiment-name",
        default=taluy_velocity_ppo_runner_cfg().experiment_name,
        help="Experiment folder under logs/rsl_rl/ when --checkpoint-file is omitted.",
    )
    parser.add_argument(
        "--run-dir",
        default=".*",
        help="Regex for the run directory when resolving from logs.",
    )
    parser.add_argument(
        "--checkpoint",
        default="model_.*.pt",
        help="Regex for the checkpoint filename when resolving from logs.",
    )
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="viser",
        help="Viewer backend.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Execution device.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel env count. Viewer inspection is usually best with 1.",
    )
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=300.0,
        help="Episode horizon for play mode.",
    )
    parser.add_argument(
        "--sampled-commands",
        action="store_true",
        help="Use task-style random command resampling instead of GUI-ready zero commands.",
    )
    parser.add_argument(
        "--fixed-command",
        type=float,
        nargs=6,
        default=None,
        metavar=("VX", "VY", "VZ", "WX", "WY", "WZ"),
        help="Optional constant body-velocity command applied during play.",
    )
    parser.add_argument(
        "--print-period-s",
        type=float,
        default=1.0,
        help="Live print period. Set <=0 to disable console prints.",
    )
    parser.add_argument(
        "--no-terminations",
        action="store_true",
        help="Disable all terminations for uninterrupted inspection.",
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
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_file is not None:
        checkpoint_path = Path(args.checkpoint_file)
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path.resolve()

    log_root = ROOT / "logs" / "rsl_rl" / str(args.experiment_name)
    try:
        return get_checkpoint_path(log_root, args.run_dir, args.checkpoint)
    except Exception as exc:
        raise SystemExit(
            f"Failed to resolve checkpoint from {log_root}: {exc}"
        ) from exc


def _build_viser_server(host: str, port: int):
    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    try:
        return viser.ViserServer(host=host, port=port, label="taluy-velocity-play")
    except TypeError:
        print("Warning: this viser version does not support host/port arguments.")
        return viser.ViserServer(label="taluy-velocity-play")


def _load_agent_cfg_dict(checkpoint_path: Path) -> dict[str, Any]:
    params_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not params_path.exists():
        return asdict(taluy_velocity_ppo_runner_cfg())

    with params_path.open("r", encoding="utf-8") as file:
        cfg = yaml.full_load(file)
    if not isinstance(cfg, dict):
        raise SystemExit(
            f"Expected mapping in {params_path}, got {type(cfg).__name__}."
        )
    return cfg


def _run_viser_viewer(
    env: RslRlVecEnvWrapper,
    policy: Callable[[Any], torch.Tensor],
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


def _make_play_env_cfg(args: argparse.Namespace):
    if args.sampled_commands:
        cfg = make_taluy_velocity_env_cfg(
            num_envs=args.num_envs,
            episode_length_s=args.episode_length_s,
        )
    else:
        cfg = make_taluy_velocity_env_cfg(
            num_envs=args.num_envs,
            episode_length_s=args.episode_length_s,
            command_resampling_time_s=(1.0e6, 1.0e6),
            command_rel_zero_envs=1.0,
            command_init_velocity_prob=0.0,
        )

    if args.no_terminations:
        cfg.terminations = {}
    return cfg


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


class InspectablePolicy:
    def __init__(
        self,
        base_policy: Callable[[Any], torch.Tensor],
        base_env: ManagerBasedRlEnv,
        print_period_steps: int,
    ) -> None:
        self._base_policy = base_policy
        self._base_env = base_env
        self._print_period_steps = max(int(print_period_steps), 1)
        self._step = 0

    def __call__(self, obs: Any) -> torch.Tensor:
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


def _run_dry_steps(
    env: RslRlVecEnvWrapper,
    policy: Callable[[Any], torch.Tensor],
    *,
    num_steps: int,
) -> None:
    obs = env.get_observations()
    reward = torch.zeros(env.num_envs, device=env.unwrapped.device)
    for _ in range(num_steps):
        with torch.no_grad():
            actions = policy(obs)
        obs, reward, _, _ = env.step(actions)

    robot = env.unwrapped.scene["robot"]
    command = env.unwrapped.command_manager.get_command("body_velocity")
    assert command is not None
    print("Dry-run complete.")
    print(f"  final_command_b={command[0].detach().cpu().tolist()}")
    print(
        f"  final_lin_vel_b={robot.data.root_link_lin_vel_b[0].detach().cpu().tolist()}"
    )
    print(
        f"  final_ang_vel_b={robot.data.root_link_ang_vel_b[0].detach().cpu().tolist()}"
    )
    print(f"  reward={reward.detach().cpu().tolist()}")


def main() -> None:
    args = _parse_args()
    device = _resolve_device(args.device)
    viewer = _resolve_viewer(args.viewer)
    checkpoint_path = _resolve_checkpoint_path(args)

    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if args.episode_length_s <= 0.0:
        raise SystemExit("--episode-length-s must be positive.")
    if args.dry_run_steps < 0:
        raise SystemExit("--dry-run-steps must be >= 0.")

    configure_torch_backends()
    os.environ.setdefault("MUJOCO_GL", "egl")

    base_env = ManagerBasedRlEnv(cfg=_make_play_env_cfg(args), device=device)
    agent_cfg_dict = _load_agent_cfg_dict(checkpoint_path)
    env = RslRlVecEnvWrapper(
        env=base_env, clip_actions=agent_cfg_dict.get("clip_actions")
    )

    try:
        runner = MjlabOnPolicyRunner(env, agent_cfg_dict, device=device)
        runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        learned_policy = runner.get_inference_policy(device=device)

        env.reset()
        if args.fixed_command is not None:
            _set_fixed_command(
                base_env, tuple(float(value) for value in args.fixed_command)
            )

        if args.print_period_s > 0.0:
            print_period_steps = max(
                int(round(args.print_period_s / base_env.step_dt)), 1
            )
            policy: Callable[[Any], torch.Tensor] = InspectablePolicy(
                learned_policy,
                base_env,
                print_period_steps=print_period_steps,
            )
        else:
            policy = cast(Callable[[Any], torch.Tensor], learned_policy)

        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"device={device} viewer={viewer} num_envs={args.num_envs}")
        if args.fixed_command is not None:
            print(f"Fixed command: {list(args.fixed_command)}")
        elif args.sampled_commands:
            print("Command mode: sampled task commands")
        else:
            print(
                "Command mode: GUI/manual. Use `Commands -> Body Velocity` in Viser "
                "to set the 6D reference."
            )

        if args.dry_run_steps > 0:
            _run_dry_steps(env, learned_policy, num_steps=args.dry_run_steps)
            return

        if viewer == "native":
            NativeMujocoViewer(env, cast(Any, policy)).run()
        else:
            _run_viser_viewer(env, policy, args.viser_host, args.viser_port)
    finally:
        env.close()


if __name__ == "__main__":
    main()
