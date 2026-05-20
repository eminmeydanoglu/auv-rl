"""Record a trained roll-policy rollout for real-time model replay."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
import math
import os
from pathlib import Path
import subprocess
from typing import Any, cast

import numpy as np

try:
    import torch
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'torch'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    import yaml
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing dependency 'pyyaml'. Install project deps first (for example `uv sync`)."
    ) from exc

try:
    from mjlab.envs import ManagerBasedRlEnv  # type: ignore[import-not-found]
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper  # type: ignore[import-not-found]
    from mjlab.utils.os import get_checkpoint_path  # type: ignore[import-not-found]
    from mjlab.utils.torch import configure_torch_backends  # type: ignore[import-not-found]
except ModuleNotFoundError as exc:
    raise SystemExit("Could not import mjlab dependencies. Ensure mjlab is available.") from exc

from auvrl import (  # noqa: E402
    ROLL_CURRICULUM_STAGES,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.actuator.body_wrench_action import BodyWrenchAction  # noqa: E402
from auvrl.actuator.thruster_actuator import THRUSTER_LOCAL_AXIS  # noqa: E402
from auvrl.tasks.roll.runtime import (  # noqa: E402
    current_root_ang_vel_b_from_qvel,
    current_root_pose_from_qpos,
    get_roll_task_state,
    quat_wxyz_to_roll_pitch_yaw,
)

from ._model_replay_io import (  # noqa: E402
    DONE_REASON_CODES,
    FORMAT_VERSION,
    done_reason_code,
    write_recording,
)

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_EXPERIMENT_NAME = taluy_roll_ppo_runner_cfg().experiment_name
DEFAULT_OUTPUT_ROOT = ROOT / "logs" / "model_replay"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-file", type=Path, default=None)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-dir", default=".*")
    parser.add_argument("--checkpoint", default="model_.*.pt")
    parser.add_argument(
        "--policy",
        choices=("checkpoint", "zero"),
        default="checkpoint",
        help="Use a trained checkpoint or a zero-action smoke policy.",
    )
    parser.add_argument(
        "--curriculum-stage",
        choices=tuple(ROLL_CURRICULUM_STAGES),
        default=None,
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--record-env-idx", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--episode-length-s", type=float, default=None)
    parser.add_argument("--roll-direction", type=int, choices=(-1, 1), default=1)
    parser.add_argument(
        "--play-mode",
        choices=("deployment", "training"),
        default="deployment",
        help="Deployment disables env terminations and records passive completion.",
    )
    parser.add_argument(
        "--deployment-completion-action",
        choices=("zero", "continue"),
        default="zero",
        help="After passive completion in deployment mode, zero or continue actions.",
    )
    parser.add_argument(
        "--post-done-steps",
        type=int,
        default=1250,
        help="Extra frames to record after passive completion.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Recording directory. Defaults to logs/model_replay/<timestamp>.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--print-period",
        type=int,
        default=250,
        help="Print progress every N recorded frames. Set <=0 to disable.",
    )
    return parser.parse_args()


def _resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Requested --device cuda, but CUDA is not available.")
    return device_arg


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path | None:
    if args.policy == "zero":
        return None
    if args.checkpoint_file is not None:
        checkpoint_path = args.checkpoint_file
        if not checkpoint_path.exists():
            raise SystemExit(f"Checkpoint file not found: {checkpoint_path}")
        return checkpoint_path.resolve()
    log_root = ROOT / "logs" / "rsl_rl" / str(args.experiment_name)
    try:
        return Path(get_checkpoint_path(log_root, args.run_dir, args.checkpoint)).resolve()
    except Exception as exc:
        raise SystemExit(f"Failed to resolve checkpoint from {log_root}: {exc}") from exc


def _load_agent_cfg_dict(checkpoint_path: Path | None) -> dict[str, Any]:
    if checkpoint_path is None:
        return asdict(taluy_roll_ppo_runner_cfg())
    params_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not params_path.exists():
        return asdict(taluy_roll_ppo_runner_cfg())
    with params_path.open("r", encoding="utf-8") as file:
        cfg = yaml.full_load(file)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Expected mapping in {params_path}, got {type(cfg).__name__}.")
    return cfg


def _default_output_dir(args: argparse.Namespace, checkpoint_path: Path | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    stage = args.curriculum_stage or "default"
    policy = checkpoint_path.stem if checkpoint_path is not None else "zero"
    return DEFAULT_OUTPUT_ROOT / f"{stamp}_{stage}_{policy}"


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _make_monitor_params(args: argparse.Namespace) -> dict[str, Any]:
    cfg = make_taluy_roll_env_cfg(
        num_envs=1,
        curriculum_stage=args.curriculum_stage,
        episode_length_s=args.episode_length_s,
        roll_direction=args.roll_direction,
    )
    task_success = cfg.terminations["task_success"].params
    return {
        "episode_length_s": float(cfg.episode_length_s),
        "target_roll_rad": float(task_success["target_roll_rad"]),
        "roll_direction": int(task_success["roll_direction"]),
        "settle_steps": int(task_success["settle_steps"]),
        "settle_pitch_limit_rad": float(task_success["settle_pitch_limit_rad"]),
        "settle_yaw_limit_rad": float(task_success["settle_yaw_limit_rad"]),
        "settle_ang_vel_limit_rad_s": float(task_success["settle_ang_vel_limit_rad_s"]),
        "settle_depth_error_limit_m": float(task_success["settle_depth_error_limit_m"]),
        "excess_pitch_limit_rad": float(cfg.terminations["excess_pitch"].params["limit_rad"]),
        "excess_depth_error_limit_m": float(
            cfg.terminations["excess_depth_error"].params["limit_m"]
        ),
        "excess_xy_drift_limit_m": float(
            cfg.terminations["excess_xy_drift"].params["limit_m"]
        ),
    }


def _make_env(args: argparse.Namespace, device: str) -> ManagerBasedRlEnv:
    cfg = make_taluy_roll_env_cfg(
        num_envs=args.num_envs,
        curriculum_stage=args.curriculum_stage,
        episode_length_s=args.episode_length_s,
        roll_direction=args.roll_direction,
    )
    if args.play_mode == "deployment":
        cfg.terminations = {}
    return ManagerBasedRlEnv(cfg=cfg, device=device)


def _wrap_angle_rad(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _tensor_to_numpy(value: Any, env_idx: int | None = None) -> np.ndarray:
    if env_idx is not None:
        value = value[env_idx]
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    else:
        value = np.asarray(value)
    return np.asarray(value)


def _env_value(value: torch.Tensor, env_idx: int) -> Any:
    selected = value[env_idx]
    if selected.ndim == 0:
        item = selected.detach().cpu().item()
        if isinstance(item, bool):
            return bool(item)
        if isinstance(item, int):
            return int(item)
        return float(item)
    return selected.detach().cpu().tolist()


def _iterable_terms_to_dict(
    terms: Sequence[tuple[str, Sequence[float]]],
    *,
    prefix: str | None = None,
) -> dict[str, list[float]]:
    result: dict[str, list[float]] = {}
    for name, values in terms:
        if prefix is not None and not name.startswith(prefix):
            continue
        key = name if prefix is None else name[len(prefix) :]
        result[key] = [float(value) for value in values]
    return result


def _active_term_dict(manager: Any, env_idx: int) -> dict[str, list[float]]:
    if not hasattr(manager, "get_active_iterable_terms"):
        return {}
    return _iterable_terms_to_dict(manager.get_active_iterable_terms(env_idx))


class PassiveTaskMonitor:
    """Deployment-style task completion monitor that does not reset the env."""

    def __init__(self, params: dict[str, Any], step_dt_s: float) -> None:
        self._params = params
        self._step_dt_s = float(step_dt_s)
        self.reset()

    def reset(self) -> None:
        self.target_reached = False
        self.settle_counter_steps = 0
        self.last_episode_step: int | None = None
        self.last: dict[str, Any] | None = None

    def update(
        self,
        *,
        env: ManagerBasedRlEnv,
        env_idx: int,
        root_pos_w: list[float],
        rpy_rad: list[float],
    ) -> dict[str, Any]:
        state = get_roll_task_state(env)
        robot = env.scene["robot"]
        episode_step = int(env.episode_length_buf[env_idx].item())
        if self.last_episode_step is not None and episode_step < self.last_episode_step:
            self.reset()

        params = self._params
        phi_total_rad = float(state.phi_total_rad[env_idx].item())
        target_roll_rad = float(params["target_roll_rad"])
        roll_direction = int(params["roll_direction"])
        signed_phi_rad = float(roll_direction) * phi_total_rad
        target_crossed_now = signed_phi_rad >= target_roll_rad
        self.target_reached = self.target_reached or target_crossed_now

        xy_ref_w = state.xy_ref_w[env_idx].detach().cpu().tolist()
        xy_drift_m = math.hypot(
            root_pos_w[0] - float(xy_ref_w[0]),
            root_pos_w[1] - float(xy_ref_w[1]),
        )
        depth_error_m = root_pos_w[2] - float(state.z_ref_m[env_idx].item())
        yaw_error_rad = _wrap_angle_rad(
            rpy_rad[2] - float(state.psi_ref_rad[env_idx].item())
        )
        ang_vel_b = current_root_ang_vel_b_from_qvel(robot)[env_idx]
        ang_speed_rad_s = float(torch.linalg.vector_norm(ang_vel_b).item())

        settle_mask = (
            abs(rpy_rad[1]) <= float(params["settle_pitch_limit_rad"])
            and abs(yaw_error_rad) <= float(params["settle_yaw_limit_rad"])
            and ang_speed_rad_s <= float(params["settle_ang_vel_limit_rad_s"])
            and abs(depth_error_m) <= float(params["settle_depth_error_limit_m"])
        )
        if self.target_reached and settle_mask:
            self.settle_counter_steps += 1
        else:
            self.settle_counter_steps = 0

        would_success = self.settle_counter_steps >= int(params["settle_steps"])
        would_fail_pitch = abs(rpy_rad[1]) > float(params["excess_pitch_limit_rad"])
        would_fail_depth = abs(depth_error_m) > float(
            params["excess_depth_error_limit_m"]
        )
        would_fail_xy = xy_drift_m > float(params["excess_xy_drift_limit_m"])
        elapsed_episode_s = episode_step * self._step_dt_s
        would_timeout = elapsed_episode_s >= float(params["episode_length_s"])
        done_reasons = [
            name
            for name, active in (
                ("task_success", would_success),
                ("excess_pitch", would_fail_pitch),
                ("excess_depth_error", would_fail_depth),
                ("excess_xy_drift", would_fail_xy),
                ("time_out", would_timeout),
            )
            if active
        ]
        monitor = {
            "progress_ratio": signed_phi_rad / target_roll_rad,
            "target_crossed_now": target_crossed_now,
            "target_reached_sticky": self.target_reached,
            "settle_mask": settle_mask,
            "settle_counter_steps": self.settle_counter_steps,
            "settle_counter_s": self.settle_counter_steps * self._step_dt_s,
            "would_success": would_success,
            "would_failure": would_fail_pitch or would_fail_depth or would_fail_xy,
            "would_timeout": would_timeout,
            "would_done": bool(done_reasons),
            "would_done_reasons": done_reasons,
            "episode_step": episode_step,
            "elapsed_episode_s": elapsed_episode_s,
            "episode_length_s": float(params["episode_length_s"]),
            "xy_drift_m": xy_drift_m,
            "depth_error_m": depth_error_m,
            "pitch_abs_rad": abs(rpy_rad[1]),
            "yaw_error_abs_rad": abs(yaw_error_rad),
            "ang_speed_rad_s": ang_speed_rad_s,
        }
        self.last_episode_step = episode_step
        self.last = monitor
        return monitor


class EpisodeRecorder:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        *,
        env_idx: int,
        policy_name: str,
        checkpoint_path: Path | None,
        curriculum_stage: str | None,
        play_mode: str,
        deployment_completion_action: str,
        monitor_params: dict[str, Any],
    ) -> None:
        self.env = env
        self.env_idx = env_idx
        self.policy_name = policy_name
        self.checkpoint_path = checkpoint_path
        self.curriculum_stage = curriculum_stage
        self.play_mode = play_mode
        self.deployment_completion_action = deployment_completion_action
        self.monitor = PassiveTaskMonitor(monitor_params, env.step_dt)
        self.controller_stopped = False
        self.controller_stop_step = -1
        self.last_action_zeroed = False
        self.frames: list[dict[str, Any]] = []

    def reset_episode(self) -> None:
        self.monitor.reset()
        self.controller_stopped = False
        self.controller_stop_step = -1
        self.last_action_zeroed = False

    def filter_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if (
            self.play_mode == "deployment"
            and self.deployment_completion_action == "zero"
            and self.controller_stopped
        ):
            self.last_action_zeroed = True
            return torch.zeros_like(actions)
        self.last_action_zeroed = False
        return actions

    def capture(
        self,
        *,
        episode_idx: int,
        action: torch.Tensor,
        obs: dict[str, torch.Tensor],
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> dict[str, Any]:
        env = self.env
        env_idx = self.env_idx
        robot = env.scene["robot"]
        root_pos_w, quat_wxyz = current_root_pose_from_qpos(robot)
        roll, pitch, yaw = quat_wxyz_to_roll_pitch_yaw(quat_wxyz)
        rpy_rad = [
            float(roll[env_idx].item()),
            float(pitch[env_idx].item()),
            float(yaw[env_idx].item()),
        ]
        root_pos = root_pos_w[env_idx].detach().cpu().tolist()
        monitor = self.monitor.update(env=env, env_idx=env_idx, root_pos_w=root_pos, rpy_rad=rpy_rad)
        if (
            self.play_mode == "deployment"
            and self.deployment_completion_action == "zero"
            and not self.controller_stopped
            and monitor["would_done"]
        ):
            self.controller_stopped = True
            self.controller_stop_step = int(monitor["episode_step"])

        wrench_term = env.action_manager.get_term("body_wrench")
        if not isinstance(wrench_term, BodyWrenchAction):
            raise RuntimeError("Expected body_wrench action term.")
        state = get_roll_task_state(env)
        reward_terms = _active_term_dict(env.reward_manager, env_idx)
        termination_terms = _active_term_dict(env.termination_manager, env_idx)
        site_ids = wrench_term.site_ids
        site_xpos = robot.data.data.site_xpos[env_idx, site_ids]
        site_xmat = robot.data.data.site_xmat[env_idx, site_ids]
        local_force_axis = torch.as_tensor(
            THRUSTER_LOCAL_AXIS,
            device=env.device,
            dtype=torch.float,
        )
        site_force_axis_w = torch.matmul(site_xmat, local_force_axis)
        done_reasons = (
            list(monitor["would_done_reasons"])
            if self.play_mode == "deployment"
            else [name for name, values in termination_terms.items() if any(values)]
        )
        done_after_step = bool(monitor["would_done"]) if self.play_mode == "deployment" else bool(
            terminated[env_idx].item() or truncated[env_idx].item()
        )
        qvel = _tensor_to_numpy(robot.data.data.qvel, env_idx).astype(np.float32)
        qpos = _tensor_to_numpy(robot.data.data.qpos, env_idx).astype(np.float32)
        root_ang_vel_b = current_root_ang_vel_b_from_qvel(robot)[env_idx]
        frame = {
            "episode_idx": episode_idx,
            "step_idx": int(env.common_step_counter),
            "episode_step": int(env.episode_length_buf[env_idx].item()),
            "time_s": float(env.episode_length_buf[env_idx].item()) * float(env.step_dt),
            "done_after_step": done_after_step,
            "would_done_reason_code": done_reason_code(done_reasons),
            "controller_stopped": self.controller_stopped,
            "controller_last_action_zeroed": self.last_action_zeroed,
            "controller_stop_step": self.controller_stop_step,
            "qpos": qpos,
            "qvel": qvel,
            "body_xpos": _tensor_to_numpy(robot.data.data.xpos, env_idx).astype(np.float32),
            "body_xmat": _tensor_to_numpy(robot.data.data.xmat, env_idx).astype(np.float32),
            "ctrl": _tensor_to_numpy(robot.data.data.ctrl, env_idx).astype(np.float32),
            "root_pos_w": np.asarray(root_pos, dtype=np.float32),
            "quat_wxyz": quat_wxyz[env_idx].detach().cpu().numpy().astype(np.float32),
            "rpy_rad": np.asarray(rpy_rad, dtype=np.float32),
            "lin_vel_b": robot.data.root_link_lin_vel_b[env_idx].detach().cpu().numpy().astype(np.float32),
            "ang_vel_b": root_ang_vel_b.detach().cpu().numpy().astype(np.float32),
            "actor_obs": obs["actor"][env_idx].detach().cpu().numpy().astype(np.float32),
            "critic_obs": obs["critic"][env_idx].detach().cpu().numpy().astype(np.float32),
            "action": action[env_idx].detach().cpu().numpy().astype(np.float32),
            "policy_wrench_b": wrench_term.action_to_wrench(action)[env_idx].detach().cpu().numpy().astype(np.float32),
            "last_applied_wrench_b": wrench_term.desired_wrench_b[env_idx].detach().cpu().numpy().astype(np.float32),
            "last_applied_wrench_origin_b": wrench_term.applied_wrench_origin_b[env_idx].detach().cpu().numpy().astype(np.float32),
            "thruster_targets_n": wrench_term.thruster_targets[env_idx].detach().cpu().numpy().astype(np.float32),
            "thruster_origin_w": site_xpos.detach().cpu().numpy().astype(np.float32),
            "thruster_force_axis_w": site_force_axis_w.detach().cpu().numpy().astype(np.float32),
            "thruster_saturation_fraction": float(wrench_term.step_saturation_fraction[env_idx].item()),
            "reward_terms": np.asarray(
                [reward_terms.get(name, [0.0])[0] for name in env.reward_manager.active_terms],
                dtype=np.float32,
            ),
            "termination_terms": np.asarray(
                [termination_terms.get(name, [0.0])[0] for name in env.termination_manager.active_terms],
                dtype=np.bool_,
            ),
            "phi_total_rad": float(state.phi_total_rad[env_idx].item()),
            "roll_progress_ratio": float(monitor["progress_ratio"]),
            "target_reached": bool(state.target_reached[env_idx].item()),
            "settle_counter_s": float(monitor["settle_counter_s"]),
            "xy_drift_m": float(monitor["xy_drift_m"]),
            "depth_error_m": float(monitor["depth_error_m"]),
            "pitch_abs_rad": float(monitor["pitch_abs_rad"]),
            "yaw_error_abs_rad": float(monitor["yaw_error_abs_rad"]),
            "ang_speed_rad_s": float(monitor["ang_speed_rad_s"]),
            "would_done": bool(monitor["would_done"]),
            "would_success": bool(monitor["would_success"]),
            "would_failure": bool(monitor["would_failure"]),
            "would_timeout": bool(monitor["would_timeout"]),
        }
        self.frames.append(frame)
        return frame

    def arrays(self) -> dict[str, np.ndarray]:
        if not self.frames:
            raise RuntimeError("No frames were recorded.")
        result: dict[str, np.ndarray] = {}
        for key in self.frames[0]:
            result[key] = np.asarray([frame[key] for frame in self.frames])
        return result


def _make_policy(
    args: argparse.Namespace,
    env: RslRlVecEnvWrapper,
    checkpoint_path: Path | None,
    agent_cfg_dict: dict[str, Any],
    device: str,
) -> Callable[[Any], torch.Tensor]:
    if args.policy == "zero":
        action = torch.zeros(
            (env.num_envs, env.unwrapped.action_manager.total_action_dim),
            device=env.unwrapped.device,
            dtype=torch.float,
        )

        def _zero_policy(_obs: Any) -> torch.Tensor:
            return action

        return _zero_policy

    if checkpoint_path is None:
        raise RuntimeError("checkpoint policy requires a checkpoint path.")
    runner = MjlabOnPolicyRunner(env, agent_cfg_dict, device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    return cast(Callable[[Any], torch.Tensor], runner.get_inference_policy(device=device))


def step_vec_env(
    env: RslRlVecEnvWrapper,
    actions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Step an RSL-RL wrapper and normalize its 4-value return to Gymnasium style."""
    obs_raw, reward, done, extras = env.step(actions)
    obs = cast(dict[str, torch.Tensor], obs_raw)
    done_tensor = cast(torch.Tensor, done)
    truncated = torch.zeros_like(done_tensor, dtype=torch.bool)
    return obs, cast(torch.Tensor, reward), done_tensor, truncated, cast(dict[str, Any], extras)


def _episode_summary(
    recorder: EpisodeRecorder,
    *,
    episode_idx: int,
    first_frame_index: int,
) -> dict[str, Any]:
    frames = recorder.frames[first_frame_index:]
    final = frames[-1]
    first_done = next((frame for frame in frames if frame["would_done"]), None)
    return {
        "episode_idx": episode_idx,
        "frames": len(frames),
        "final_step": int(final["episode_step"]),
        "final_time_s": float(final["time_s"]),
        "first_would_done_step": None
        if first_done is None
        else int(first_done["episode_step"]),
        "would_done": bool(final["would_done"]),
        "would_success": bool(final["would_success"]),
        "would_failure": bool(final["would_failure"]),
        "would_timeout": bool(final["would_timeout"]),
        "phi_total_rad": float(final["phi_total_rad"]),
        "roll_progress_ratio": float(final["roll_progress_ratio"]),
        "xy_drift_m": float(final["xy_drift_m"]),
        "depth_error_m": float(final["depth_error_m"]),
    }


def main() -> None:
    args = _parse_args()
    if args.num_envs <= 0:
        raise SystemExit("--num-envs must be positive.")
    if not 0 <= args.record_env_idx < args.num_envs:
        raise SystemExit("--record-env-idx must select an existing env.")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive.")
    if args.post_done_steps < 0:
        raise SystemExit("--post-done-steps must be >= 0.")

    device = _resolve_device(args.device)
    checkpoint_path = _resolve_checkpoint_path(args)
    output_dir = args.output_dir or _default_output_dir(args, checkpoint_path)
    monitor_params = _make_monitor_params(args)

    os.environ.setdefault("MUJOCO_GL", "egl")
    configure_torch_backends()

    base_env = _make_env(args, device)
    agent_cfg_dict = _load_agent_cfg_dict(checkpoint_path)
    env = RslRlVecEnvWrapper(
        env=base_env,
        clip_actions=agent_cfg_dict.get("clip_actions"),
    )

    try:
        policy = _make_policy(args, env, checkpoint_path, agent_cfg_dict, device)
        recorder = EpisodeRecorder(
            base_env,
            env_idx=args.record_env_idx,
            policy_name=args.policy,
            checkpoint_path=checkpoint_path,
            curriculum_stage=args.curriculum_stage,
            play_mode=args.play_mode,
            deployment_completion_action=args.deployment_completion_action,
            monitor_params=monitor_params,
        )
        summaries: list[dict[str, Any]] = []
        default_max_steps = int(math.ceil(monitor_params["episode_length_s"] / base_env.step_dt))
        max_steps = args.max_steps or default_max_steps

        for episode_idx in range(args.episodes):
            env.reset()
            recorder.reset_episode()
            first_frame_index = len(recorder.frames)
            first_done_step: int | None = None
            for _ in range(max_steps + args.post_done_steps):
                obs = env.get_observations()
                with torch.no_grad():
                    actions = policy(obs)
                    actions = recorder.filter_actions(actions)
                obs_after, _reward, terminated, truncated, _extras = step_vec_env(
                    env, actions
                )
                frame = recorder.capture(
                    episode_idx=episode_idx,
                    action=actions,
                    obs=obs_after,
                    terminated=terminated,
                    truncated=truncated,
                )
                if frame["would_done"] and first_done_step is None:
                    first_done_step = int(frame["episode_step"])
                if args.play_mode == "training" and frame["done_after_step"]:
                    break
                if (
                    args.play_mode == "deployment"
                    and first_done_step is not None
                    and int(frame["episode_step"]) >= first_done_step + args.post_done_steps
                ):
                    break
                if args.print_period > 0 and len(recorder.frames) % args.print_period == 0:
                    print(
                        f"recorded_frames={len(recorder.frames)} "
                        f"episode={episode_idx} step={frame['episode_step']} "
                        f"progress={frame['roll_progress_ratio']:.3f} "
                        f"done={frame['would_done']}"
                    )
            summaries.append(
                _episode_summary(
                    recorder,
                    episode_idx=episode_idx,
                    first_frame_index=first_frame_index,
                )
            )

        arrays = recorder.arrays()
        manifest = {
            "format": FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "git_commit": _git_commit(),
            "checkpoint_file": str(checkpoint_path) if checkpoint_path is not None else None,
            "policy": args.policy,
            "curriculum_stage": args.curriculum_stage,
            "play_mode": args.play_mode,
            "deployment_completion_action": args.deployment_completion_action,
            "episodes": args.episodes,
            "max_steps": max_steps,
            "post_done_steps": args.post_done_steps,
            "physics_dt_s": float(base_env.cfg.sim.mujoco.timestep),
            "control_dt_s": float(base_env.step_dt),
            "decimation": int(base_env.cfg.decimation),
            "reward_term_names": list(base_env.reward_manager.active_terms),
            "termination_term_names": list(base_env.termination_manager.active_terms),
            "actor_obs_names": list(base_env.cfg.observations["actor"].terms.keys()),
            "critic_obs_names": list(base_env.cfg.observations["critic"].terms.keys()),
            "done_reason_codes": DONE_REASON_CODES,
        }
        write_recording(
            output_dir,
            manifest=manifest,
            arrays=arrays,
            episode_summaries=summaries,
            overwrite=args.overwrite,
        )
        print(f"Recording written: {output_dir.resolve()}")
        print(f"  frames={arrays['time_s'].shape[0]} episodes={len(summaries)}")
        print(f"  trajectory={output_dir.resolve() / 'trajectory.npz'}")
        print(f"  manifest={output_dir.resolve() / 'manifest.json'}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
