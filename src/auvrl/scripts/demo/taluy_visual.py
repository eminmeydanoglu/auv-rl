"""Visual Taluy MJLab demo with scripted thruster commands."""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path
from typing import Any, cast

import mujoco
import numpy as np

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
        "Could not import mjlab. Ensure mjlab is available "
        "(local ../mjlab or installed dependency)."
    ) from exc

from auvrl import make_taluy_base_env_cfg  # type: ignore[import-not-found]  # noqa: E402
from auvrl import UnderwaterHydroActionCfg  # type: ignore[import-not-found]  # noqa: E402
from auvrl.actuator.thruster_allocation import (  # type: ignore[import-not-found]  # noqa: E402
    allocation_matrix_from_mujoco_sites,
)
from auvrl.actuator.thruster_actuator import (  # type: ignore[import-not-found]  # noqa: E402
    THRUSTER_LOCAL_AXIS,
)
from auvrl.config.thruster_cfg import (  # type: ignore[import-not-found]  # noqa: E402
    THRUSTER_CFG_DIR,
    load_thruster_cfg,
)
from auvrl.config.auv_cfg import AUVMjlabCfg  # type: ignore[import-not-found]  # noqa: E402
from auvrl.config.auv_cfg import (  # type: ignore[import-not-found]  # noqa: E402
    TALUY_CFG_PATH,
    load_auv_cfg,
)


def _demo_phase_commands(taluy_cfg: AUVMjlabCfg) -> dict[str, np.ndarray]:
    return {
        "surge": np.asarray(taluy_cfg.surge_command, dtype=float),
        "yaw": np.asarray(taluy_cfg.yaw_command, dtype=float),
        "heave": np.asarray(taluy_cfg.heave_command, dtype=float),
        "coast": np.asarray(taluy_cfg.coast_command, dtype=float),
    }


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_viewer(viewer: str) -> str:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


class TaluyScriptedPolicy:
    def __init__(
        self,
        *,
        num_envs: int,
        action_dim: int,
        step_dt: float,
        device: str,
        phase: str,
        phase_commands: dict[str, np.ndarray],
        debugger: "TaluyLiveDebugger | None" = None,
    ):
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._step_dt = step_dt
        self._device = device
        self._phase = phase
        self._phase_commands = phase_commands
        self._debugger = debugger
        self._step_idx = 0

    def reset(self) -> None:
        self._step_idx = 0

    def __call__(self, obs: object) -> torch.Tensor:
        del obs
        cmd = self._phase_command()

        if self._debugger is not None:
            self._debugger.maybe_log(self._step_idx, cmd)

        self._step_idx += 1

        action = torch.zeros(
            (self._num_envs, self._action_dim), device=self._device, dtype=torch.float
        )
        action[:, :8] = torch.as_tensor(cmd, device=self._device, dtype=torch.float)
        return action

    def _phase_command(self) -> np.ndarray:
        if self._phase != "cycle":
            return self._phase_commands[self._phase]

        sim_t = self._step_idx * self._step_dt
        phase_time = sim_t % 16.0
        if phase_time < 4.0:
            return self._phase_commands["surge"]
        if phase_time < 8.0:
            return self._phase_commands["yaw"]
        if phase_time < 12.0:
            return self._phase_commands["heave"]
        return self._phase_commands["coast"]


def _apply_demo_stabilization(
    cfg: object,
    *,
    center_of_gravity_b_m: tuple[float, float, float],
    cob_z_offset_m: float,
) -> None:
    actions = getattr(cfg, "actions", {})
    hydro = actions.get("hydro")
    if hydro is None:
        return
    hydro_cfg = cast(UnderwaterHydroActionCfg, hydro)

    damping = [list(row) for row in hydro_cfg.linear_damping_matrix_6x6]
    yaw_drag = float(damping[5][5])
    damping[3][3] = 8.0
    damping[4][4] = 8.0
    damping[5][5] = yaw_drag
    hydro_cfg.linear_damping_matrix_6x6 = cast(
        Any,
        tuple(tuple(float(value) for value in row) for row in damping),
    )

    if abs(cob_z_offset_m) > 1e-12:
        cog_x, cog_y, cog_z = center_of_gravity_b_m
        hydro_cfg.center_of_buoyancy_b_m = (
            float(cog_x),
            float(cog_y),
            float(cog_z + cob_z_offset_m),
        )


def _resolve_taluy_allocation(
    base_env: ManagerBasedRlEnv,
    taluy_cfg: AUVMjlabCfg,
) -> tuple[str, str, np.ndarray, np.ndarray]:
    model = base_env.sim.mj_model
    data = base_env.sim.mj_data
    mj_name2id = getattr(mujoco, "mj_name2id")
    mjt_obj = getattr(mujoco, "mjtObj")

    entity_names = [
        name for name in base_env.scene.entities.keys() if name != "terrain"
    ]
    entity_prefix = "robot" if "robot" in entity_names else entity_names[0]

    def resolve_name(obj_type: int, base_name: str) -> str:
        if mj_name2id(model, obj_type, base_name) >= 0:
            return base_name
        prefixed = f"{entity_prefix}/{base_name}"
        if mj_name2id(model, obj_type, prefixed) >= 0:
            return prefixed
        raise ValueError(
            f"Could not find {base_name!r} (or prefixed variant) in scene model."
        )

    body_name = resolve_name(mjt_obj.mjOBJ_BODY, taluy_cfg.body_name)
    site_names = [
        resolve_name(mjt_obj.mjOBJ_SITE, site_name)
        for site_name in taluy_cfg.thruster_site_names
    ]
    allocation = allocation_matrix_from_mujoco_sites(
        model=model,
        data=data,
        body_name=body_name,
        site_names=site_names,
        local_force_axis=THRUSTER_LOCAL_AXIS,
    )
    body_id = mj_name2id(model, mjt_obj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo model")
    com_body_m = np.asarray(model.body_ipos[body_id], dtype=float).reshape(3)

    return entity_prefix, body_name, allocation, com_body_m.copy()


def _allocation_about_com(
    allocation_origin: np.ndarray,
    com_body_m: np.ndarray,
) -> np.ndarray:
    allocation_com = np.asarray(allocation_origin, dtype=float).copy()
    com = np.asarray(com_body_m, dtype=float).reshape(3)
    for index in range(allocation_com.shape[1]):
        force = allocation_com[0:3, index]
        allocation_com[3:6, index] -= np.cross(com, force)
    return allocation_com


def _solve_command_for_wrench(
    allocation: np.ndarray,
    target_wrench: np.ndarray,
    command_limit: float,
) -> np.ndarray:
    cmd, *_ = np.linalg.lstsq(allocation, target_wrench, rcond=None)
    cmd = np.asarray(cmd, dtype=float).reshape(-1)
    max_abs = float(np.max(np.abs(cmd))) if cmd.size else 0.0
    if max_abs > command_limit:
        cmd *= command_limit / max_abs
    return cmd


def _build_com_neutral_phase_commands(
    allocation_origin: np.ndarray,
    com_body_m: np.ndarray,
    reference_commands: dict[str, np.ndarray],
    command_limit: float,
) -> dict[str, np.ndarray]:
    allocation_com = _allocation_about_com(allocation_origin, com_body_m)

    surge_ref = allocation_origin @ reference_commands["surge"]
    yaw_ref = allocation_origin @ reference_commands["yaw"]
    heave_ref = allocation_origin @ reference_commands["heave"]

    target_surge = np.array([surge_ref[0], 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    target_yaw = np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw_ref[5]], dtype=float)
    target_heave = np.array([0.0, 0.0, heave_ref[2], 0.0, 0.0, 0.0], dtype=float)
    target_coast = np.zeros(6, dtype=float)

    commands = {
        "surge": _solve_command_for_wrench(allocation_com, target_surge, command_limit),
        "yaw": _solve_command_for_wrench(allocation_com, target_yaw, command_limit),
        "heave": _solve_command_for_wrench(allocation_com, target_heave, command_limit),
        "coast": _solve_command_for_wrench(allocation_com, target_coast, command_limit),
    }
    return commands


def _print_phase_wrenches(
    body_name: str,
    allocation_origin: np.ndarray,
    allocation_com: np.ndarray,
    phase_commands: dict[str, np.ndarray],
) -> None:
    phases = ("surge", "yaw", "heave", "coast")
    print("Phase wrench check [Fx, Fy, Fz, Mx, My, Mz]")
    print(f"  using body={body_name}")
    for name in phases:
        cmd = phase_commands[name]

        wrench_origin = allocation_origin @ cmd
        wrench_com = allocation_com @ cmd
        print(
            f"  {name:>5}: cmd={np.round(cmd, 3)} "
            f"origin={np.round(wrench_origin, 3)} "
            f"com={np.round(wrench_com, 3)}"
        )


def _quat_wxyz_to_euler_deg(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = [float(v) for v in quat_wxyz]

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return (float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw)))


class TaluyLiveDebugger:
    def __init__(
        self,
        *,
        base_env: ManagerBasedRlEnv,
        entity_name: str,
        body_name: str,
        allocation_origin: np.ndarray,
        allocation_com: np.ndarray,
        period_s: float,
    ) -> None:
        self._entity = base_env.scene[entity_name]
        body_ids, _ = self._entity.find_bodies(body_name)
        if len(body_ids) != 1:
            raise ValueError(
                f"Expected one Taluy body match, got {body_ids} for {entity_name}."
            )
        self._body_id = int(body_ids[0])

        self._allocation_origin = np.asarray(allocation_origin, dtype=float)
        self._allocation_com = np.asarray(allocation_com, dtype=float)
        self._period_steps = max(1, int(round(period_s / max(base_env.step_dt, 1e-6))))
        self._actuator = self._entity.actuators[0] if self._entity.actuators else None

    def maybe_log(self, step_idx: int, cmd: np.ndarray) -> None:
        if step_idx % self._period_steps != 0:
            return

        quat = (
            self._entity.data.body_link_quat_w[0, self._body_id].detach().cpu().numpy()
        )
        lin_vel = (
            self._entity.data.body_link_lin_vel_w[0, self._body_id]
            .detach()
            .cpu()
            .numpy()
        )
        ang_vel = (
            self._entity.data.body_link_ang_vel_w[0, self._body_id]
            .detach()
            .cpu()
            .numpy()
        )
        rpy_deg = _quat_wxyz_to_euler_deg(quat)

        wrench_cmd_origin = self._allocation_origin @ np.asarray(cmd, dtype=float)
        wrench_cmd_com = self._allocation_com @ np.asarray(cmd, dtype=float)

        thrust_state = None
        if self._actuator is not None:
            thrust_state = getattr(self._actuator, "_thrust_state", None)

        if thrust_state is not None and hasattr(thrust_state, "detach"):
            thrust_np = thrust_state[0].detach().cpu().numpy()
            wrench_act_origin = self._allocation_origin @ thrust_np
            wrench_act_com = self._allocation_com @ thrust_np
            print(
                "[live] "
                f"rpy_deg={np.round(rpy_deg, 2)} "
                f"lin={np.round(lin_vel, 3)} "
                f"ang={np.round(ang_vel, 3)} "
                f"w_org_cmd={np.round(wrench_cmd_origin, 3)} "
                f"w_com_cmd={np.round(wrench_cmd_com, 3)} "
                f"w_org_act={np.round(wrench_act_origin, 3)} "
                f"w_com_act={np.round(wrench_act_com, 3)}"
            )
            return

        print(
            "[live] "
            f"rpy_deg={np.round(rpy_deg, 2)} "
            f"lin={np.round(lin_vel, 3)} "
            f"ang={np.round(ang_vel, 3)} "
            f"w_org_cmd={np.round(wrench_cmd_origin, 3)} "
            f"w_com_cmd={np.round(wrench_cmd_com, 3)}"
        )


def _build_viser_server(host: str, port: int):
    try:
        import viser  # type: ignore[import-not-found]
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependency 'viser'.") from exc

    try:
        return viser.ViserServer(host=host, port=port, label="taluy-mjlab")
    except TypeError:
        print("Warning: this viser version does not support host/port arguments.")
        return viser.ViserServer(label="taluy-mjlab")


def _run_viser_viewer(
    env: RslRlVecEnvWrapper,
    policy: TaluyScriptedPolicy,
    host: str,
    port: int,
) -> None:
    params = inspect.signature(ViserPlayViewer.__init__).parameters

    if "viser_server" in params:
        server = _build_viser_server(host, port)
        ViserPlayViewer(env, policy, viser_server=server).run()
        return

    if "server" in params:
        server = _build_viser_server(host, port)
        ViserPlayViewer(env, policy, server=server).run()  # type: ignore[call-arg]
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
        ViserPlayViewer(env, policy).run()
    finally:
        viser.ViserServer = original_server_cls


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--viewer",
        choices=("auto", "native", "viser"),
        default="auto",
        help="Viewer backend.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device.",
    )
    parser.add_argument(
        "--phase",
        choices=("cycle", "surge", "yaw", "heave", "coast"),
        default="cycle",
        help="Hold a single motion phase or run full cycle.",
    )
    parser.add_argument(
        "--command-profile",
        choices=("legacy", "com-neutral"),
        default="com-neutral",
        help="Thruster command profile for scripted debug/demo only (not RL training).",
    )
    parser.add_argument(
        "--stabilize",
        action="store_true",
        help="Apply extra roll/pitch damping for easier visual debugging.",
    )
    parser.add_argument(
        "--cob-z-offset",
        type=float,
        default=0.0,
        help=(
            "Optional CoB offset from CoG along body z (m). "
            "Use with care; sign depends on body frame convention."
        ),
    )
    parser.add_argument(
        "--debug-wrench",
        action="store_true",
        help="Print net phase wrench from current Taluy thruster allocation.",
    )
    parser.add_argument(
        "--debug-live",
        action="store_true",
        help="Print live attitude, rates, and commanded/actual wrench.",
    )
    parser.add_argument(
        "--debug-period-s",
        type=float,
        default=1.0,
        help="Live debug print period in seconds.",
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
        help="Viser TCP port.",
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    viewer = _resolve_viewer(args.viewer)
    taluy_cfg = load_auv_cfg(TALUY_CFG_PATH)

    cfg = make_taluy_base_env_cfg(
        action_space="thruster",
    )
    if args.stabilize:
        _apply_demo_stabilization(
            cfg,
            center_of_gravity_b_m=taluy_cfg.center_of_gravity_b_m,
            cob_z_offset_m=args.cob_z_offset,
        )
    cfg.viewer.distance = 3.5
    cfg.viewer.elevation = -20.0
    cfg.viewer.azimuth = 130.0

    base_env = ManagerBasedRlEnv(cfg=cfg, device=device)

    entity_name, body_name, allocation_origin, com_body_m = _resolve_taluy_allocation(
        base_env,
        taluy_cfg,
    )
    allocation_com = _allocation_about_com(allocation_origin, com_body_m)

    phase_commands = _demo_phase_commands(taluy_cfg)
    if args.command_profile == "com-neutral":
        command_limit = float(
            load_thruster_cfg(
                THRUSTER_CFG_DIR / f"{taluy_cfg.thruster_model}.yaml"
            ).command_limit
        )
        phase_commands = _build_com_neutral_phase_commands(
            allocation_origin,
            com_body_m,
            phase_commands,
            command_limit,
        )

    if args.debug_wrench:
        _print_phase_wrenches(
            body_name,
            allocation_origin,
            allocation_com,
            phase_commands,
        )

    debugger = None
    if args.debug_live:
        debugger = TaluyLiveDebugger(
            base_env=base_env,
            entity_name=entity_name,
            body_name=taluy_cfg.body_name,
            allocation_origin=allocation_origin,
            allocation_com=allocation_com,
            period_s=max(float(args.debug_period_s), 0.05),
        )

    env = RslRlVecEnvWrapper(base_env)
    policy = TaluyScriptedPolicy(
        num_envs=env.num_envs,
        action_dim=env.unwrapped.action_manager.total_action_dim,
        step_dt=env.unwrapped.step_dt,
        device=device,
        phase=args.phase,
        phase_commands=phase_commands,
        debugger=debugger,
    )

    print(f"Taluy MJLab visual demo | device={device} viewer={viewer}")
    if args.phase == "cycle":
        print("Script: 16s cycle (surge -> yaw -> heave -> coast)")
    else:
        print(f"Script: fixed phase '{args.phase}'")
    print(f"Command profile: {args.command_profile}")

    if viewer == "native":
        NativeMujocoViewer(env, policy).run()
    else:
        _run_viser_viewer(env, policy, args.viser_host, args.viser_port)

    env.close()


if __name__ == "__main__":
    main()
