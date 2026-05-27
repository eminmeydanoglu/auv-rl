from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.curriculum_manager import CurriculumTermCfg

from auvrl.tasks.roll.curriculum import RollCurriculumStage

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


POST_C3L_POLISH_AUTO_CURRICULUM = "post_c3l_polish"


@dataclass(frozen=True)
class PostC3LPolishSchedule:
    start_stage: RollCurriculumStage
    goal_stage: RollCurriculumStage
    rolling_window_episodes: int = 4096
    min_completed_episodes_per_update: int = 512
    observe_updates: int = 2
    success_advance_threshold: float = 0.98
    success_rollback_threshold: float = 0.95
    xy_peak_advance_max_m: float = 0.35
    xy_peak_rollback_max_m: float = 0.50
    pitch_peak_advance_max_deg: float = 72.0
    pitch_peak_rollback_max_deg: float = 78.0
    saturation_rollback_max: float = 0.75
    action_l2_rollback_max: float = 6.5
    k_xy_step: float = 0.005
    k_pitch_step: float = 0.025
    k_yaw_step: float = 0.01
    k_depth_step: float = 0.01
    settle_window_s_step: float = 0.01
    settle_pitch_limit_deg_step: float = 1.0
    settle_yaw_limit_deg_step: float = 2.0
    settle_ang_vel_limit_rad_s_step: float = 0.05
    settle_depth_error_limit_m_step: float = 0.05
    settle_xy_drift_limit_m_step: float = 0.02
    excess_pitch_deg_step: float = 1.0
    k_smooth_step: float = 0.0005
    k_action_effort_step: float = 0.00015
    k_thruster_saturation_step: float = 0.005
    thruster_saturation_threshold_step: float = 0.01


class PostC3LPolishCurriculum:
    _PHASES = (
        "observe",
        "attitude_depth",
        "hard_pitch_envelope",
        "settle_attitude",
        "settle_window",
        "settle_ang_vel",
        "settle_depth",
        "settle_xy",
        "smoothness",
        "saturation_weight",
        "saturation_threshold",
        "done",
    )

    _REWARD_FIELDS = {
        "k_xy": ("xy_drift", 1.0),
        "k_pitch": ("pitch_penalty", 1.0),
        "k_yaw": ("yaw_hold", 1.0),
        "k_depth": ("depth_hold", 1.0),
        "k_smooth": ("action_smoothness", -1.0),
        "k_action_effort": ("action_effort", -1.0),
        "k_thruster_saturation": ("thruster_saturation", -1.0),
    }

    _TERMINATION_FIELDS = {
        "settle_pitch_limit_deg": "settle_pitch_limit_rad",
        "settle_yaw_limit_deg": "settle_yaw_limit_rad",
        "settle_ang_vel_limit_rad_s": "settle_ang_vel_limit_rad_s",
        "settle_depth_error_limit_m": "settle_depth_error_limit_m",
        "settle_xy_drift_limit_m": "settle_xy_drift_limit_m",
    }

    _PHASE_FIELDS = {
        "attitude_depth": (
            "k_xy",
            "k_pitch",
            "k_yaw",
            "k_depth",
        ),
        "hard_pitch_envelope": ("excess_pitch_deg",),
        "settle_attitude": (
            "settle_pitch_limit_deg",
            "settle_yaw_limit_deg",
        ),
        "settle_window": ("settle_window_s",),
        "settle_ang_vel": ("settle_ang_vel_limit_rad_s",),
        "settle_depth": ("settle_depth_error_limit_m",),
        "settle_xy": ("settle_xy_drift_limit_m",),
        "smoothness": (
            "k_smooth",
            "k_action_effort",
        ),
        "saturation_weight": ("k_thruster_saturation",),
        "saturation_threshold": ("thruster_saturation_threshold",),
    }

    _STEP_FIELDS = {
        "k_xy": "k_xy_step",
        "k_pitch": "k_pitch_step",
        "k_yaw": "k_yaw_step",
        "k_depth": "k_depth_step",
        "settle_window_s": "settle_window_s_step",
        "settle_pitch_limit_deg": "settle_pitch_limit_deg_step",
        "settle_yaw_limit_deg": "settle_yaw_limit_deg_step",
        "settle_ang_vel_limit_rad_s": "settle_ang_vel_limit_rad_s_step",
        "settle_depth_error_limit_m": "settle_depth_error_limit_m_step",
        "settle_xy_drift_limit_m": "settle_xy_drift_limit_m_step",
        "excess_pitch_deg": "excess_pitch_deg_step",
        "k_smooth": "k_smooth_step",
        "k_action_effort": "k_action_effort_step",
        "k_thruster_saturation": "k_thruster_saturation_step",
        "thruster_saturation_threshold": "thruster_saturation_threshold_step",
    }

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRlEnv) -> None:
        del env
        schedule = cfg.params["schedule"]
        if not isinstance(schedule, PostC3LPolishSchedule):
            raise TypeError("schedule must be a PostC3LPolishSchedule.")
        self._schedule = schedule
        self._start = _stage_values(schedule.start_stage)
        self._goal = _stage_values(schedule.goal_stage)
        if not math.isclose(self._start["target_roll_deg"], 720.0):
            raise ValueError("post-c3l polish requires a 720 degree start stage.")
        if not math.isclose(self._goal["target_roll_deg"], 720.0):
            raise ValueError("post-c3l polish requires a 720 degree goal stage.")
        self._values = dict(self._start)
        self._phase_index = 0
        self._observe_count = 0
        self._completed_since_update = 0
        self._total_completed = 0
        self._advance_count = 0
        self._rollback_count = 0
        self._phase_update_count = 0
        self._phase_completed_episodes = 0
        self._phase_advance_count = 0
        self._phase_rollback_count = 0
        self._last_update = 0
        self._last_safe = 0
        self._last_rollback = 0
        self._success = deque(maxlen=schedule.rolling_window_episodes)
        self._target_reached = deque(maxlen=schedule.rolling_window_episodes)
        self._time_out = deque(maxlen=schedule.rolling_window_episodes)
        self._excess_pitch = deque(maxlen=schedule.rolling_window_episodes)
        self._excess_depth_error = deque(maxlen=schedule.rolling_window_episodes)
        self._excess_xy_drift = deque(maxlen=schedule.rolling_window_episodes)
        self._first_done_s = deque(maxlen=schedule.rolling_window_episodes)
        self._xy_peak_m = deque(maxlen=schedule.rolling_window_episodes)
        self._pitch_peak_rad = deque(maxlen=schedule.rolling_window_episodes)
        self._depth_abs_error_m = deque(maxlen=schedule.rolling_window_episodes)
        self._pitch_abs_rad = deque(maxlen=schedule.rolling_window_episodes)
        self._yaw_abs_error_rad = deque(maxlen=schedule.rolling_window_episodes)
        self._root_ang_speed_rad_s = deque(maxlen=schedule.rolling_window_episodes)
        self._action_l2 = deque(maxlen=schedule.rolling_window_episodes)
        self._saturation = deque(maxlen=schedule.rolling_window_episodes)

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor | slice | None,
        schedule: PostC3LPolishSchedule | None = None,
    ) -> dict[str, float]:
        del schedule
        ids = _env_id_tensor(env, env_ids)
        self._record_completed_episodes(env, ids)
        self._last_update = 0
        self._last_safe = int(self._can_advance())
        self._last_rollback = 0
        if self._completed_since_update >= self._schedule.min_completed_episodes_per_update:
            self._update_curriculum()
            self._completed_since_update = 0
        self._apply_values(env)
        return self._state()

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        del env_ids

    def _record_completed_episodes(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
    ) -> None:
        metrics_manager = env.metrics_manager
        counts = getattr(metrics_manager, "_step_count", None)
        if counts is None or env_ids.numel() == 0:
            return
        env_counts = counts[env_ids]
        valid = env_counts > 0
        if not valid.any().item():
            return
        ids = env_ids[valid]
        valid_counts = env_counts[valid]
        try:
            success = env.termination_manager.get_term("task_success")[ids].bool()
        except KeyError:
            success = torch.zeros(ids.numel(), dtype=torch.bool, device=ids.device)
        self._success.extend(float(item) for item in success.float().detach().cpu().tolist())
        self._extend_termination(env, ids, "time_out", self._time_out)
        self._extend_termination(env, ids, "excess_pitch", self._excess_pitch)
        self._extend_termination(env, ids, "excess_depth_error", self._excess_depth_error)
        self._extend_termination(env, ids, "excess_xy_drift", self._excess_xy_drift)
        if success.any().item():
            times = valid_counts[success].float() * float(env.step_dt)
            self._first_done_s.extend(float(item) for item in times.detach().cpu().tolist())
        target_reached = _episode_metric_values(env, ids, "target_reached_last")
        if target_reached is not None:
            self._target_reached.extend(
                float(item) for item in target_reached.detach().cpu().tolist()
            )
        self._extend_metric(env, ids, "xy_drift_m_peak", self._xy_peak_m)
        self._extend_metric(env, ids, "pitch_abs_peak_rad", self._pitch_peak_rad)
        self._extend_metric(env, ids, "depth_abs_error_m", self._depth_abs_error_m)
        self._extend_metric(env, ids, "pitch_abs_rad", self._pitch_abs_rad)
        self._extend_metric(env, ids, "yaw_abs_error_rad", self._yaw_abs_error_rad)
        self._extend_metric(env, ids, "root_ang_speed_rad_s", self._root_ang_speed_rad_s)
        self._extend_metric(env, ids, "body_wrench_action_l2", self._action_l2)
        self._extend_metric(env, ids, "body_wrench_saturation_fraction", self._saturation)
        completed = int(ids.numel())
        self._completed_since_update += completed
        self._total_completed += completed
        self._phase_completed_episodes += completed

    def _extend_metric(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        name: str,
        target: deque[float],
    ) -> None:
        values = _episode_metric_values(env, env_ids, name)
        if values is None:
            return
        target.extend(float(item) for item in values.detach().cpu().tolist())

    def _extend_termination(
        self,
        env: ManagerBasedRlEnv,
        env_ids: torch.Tensor,
        name: str,
        target: deque[float],
    ) -> None:
        try:
            values = env.termination_manager.get_term(name)[env_ids].float()
        except KeyError:
            return
        target.extend(float(item) for item in values.detach().cpu().tolist())

    def _update_curriculum(self) -> None:
        self._phase_update_count += 1
        if self._phase_index == 0:
            self._observe_count += 1
            if self._observe_count >= self._schedule.observe_updates and self._can_advance():
                self._set_phase_index(1)
            return
        if self._must_rollback():
            self._rollback_phase()
            return
        phase = self._PHASES[self._phase_index]
        if phase == "done":
            return
        if self._can_advance():
            self._advance_phase()

    def _advance_phase(self) -> None:
        phase = self._PHASES[self._phase_index]
        self._move_phase(phase, self._goal)
        self._advance_count += 1
        self._phase_advance_count += 1
        self._last_update = 1
        if self._phase_complete(phase, self._goal):
            self._set_phase_index(min(self._phase_index + 1, len(self._PHASES) - 1))

    def _rollback_phase(self) -> None:
        phase = self._PHASES[self._phase_index]
        if phase == "done" and self._phase_index > 1:
            self._set_phase_index(self._phase_index - 1)
            phase = self._PHASES[self._phase_index]
        changed = self._move_phase(phase, self._start)
        if not changed and self._phase_index > 1:
            self._set_phase_index(self._phase_index - 1)
            phase = self._PHASES[self._phase_index]
            self._move_phase(phase, self._start)
        self._rollback_count += 1
        self._phase_rollback_count += 1
        self._last_update = -1
        self._last_rollback = 1

    def _set_phase_index(self, phase_index: int) -> None:
        if phase_index == self._phase_index:
            return
        self._phase_index = phase_index
        self._phase_update_count = 0
        self._phase_completed_episodes = 0
        self._phase_advance_count = 0
        self._phase_rollback_count = 0

    def _move_phase(self, phase: str, target_values: dict[str, float]) -> bool:
        changed = False
        for field in self._PHASE_FIELDS[phase]:
            old = self._values[field]
            new = _move_toward(
                old,
                target_values[field],
                getattr(self._schedule, self._STEP_FIELDS[field]),
            )
            if not math.isclose(old, new, rel_tol=0.0, abs_tol=1.0e-12):
                self._values[field] = new
                changed = True
        return changed

    def _phase_complete(self, phase: str, target_values: dict[str, float]) -> bool:
        return all(
            math.isclose(
                self._values[field],
                target_values[field],
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            for field in self._PHASE_FIELDS[phase]
        )

    def _can_advance(self) -> bool:
        if not self._success:
            return False
        gates = self._gate_state()
        blockers = (
            "advance_blocked_by_success",
            "advance_blocked_by_xy",
            "advance_blocked_by_pitch",
            "advance_blocked_by_root_ang_speed",
            "advance_blocked_by_depth",
            "advance_blocked_by_settle_xy",
        )
        return not any(bool(gates[name]) for name in blockers)

    def _must_rollback(self) -> bool:
        if not self._success:
            return False
        success_rate = _mean(self._success)
        xy_peak_p95 = _quantile(self._xy_peak_m, 0.95, default=float("inf"))
        pitch_peak_p95 = math.degrees(
            _quantile(self._pitch_peak_rad, 0.95, default=float("inf"))
        )
        phase = self._PHASES[self._phase_index]
        xy_rollback_limit = self._xy_rollback_limit_for_phase(phase)
        pitch_rollback_limit = max(
            self._schedule.pitch_peak_rollback_max_deg,
            self._values["excess_pitch_deg"] + 2.0,
        )
        saturation_mean = _mean(self._saturation)
        action_l2_mean = _mean(self._action_l2)
        return (
            success_rate < self._schedule.success_rollback_threshold
            or xy_peak_p95 > xy_rollback_limit
            or pitch_peak_p95 > pitch_rollback_limit
            or saturation_mean > self._schedule.saturation_rollback_max
            or action_l2_mean > self._schedule.action_l2_rollback_max
        )

    def _apply_values(self, env: ManagerBasedRlEnv) -> None:
        for field, (term_name, sign) in self._REWARD_FIELDS.items():
            term_cfg = env.reward_manager.get_term_cfg(term_name)
            term_cfg.weight = sign * self._values[field]
        sat_cfg = env.reward_manager.get_term_cfg("thruster_saturation")
        sat_cfg.params["threshold"] = self._values["thruster_saturation_threshold"]
        task_success_cfg = env.termination_manager.get_term_cfg("task_success")
        params = task_success_cfg.params
        params["settle_steps"] = max(
            1,
            math.ceil(self._values["settle_window_s"] / float(env.step_dt)),
        )
        for field, param_name in self._TERMINATION_FIELDS.items():
            value = self._values[field]
            if field.endswith("_deg"):
                value = math.radians(value)
            params[param_name] = value
        excess_pitch_cfg = env.termination_manager.get_term_cfg("excess_pitch")
        excess_pitch_cfg.params["limit_rad"] = math.radians(
            self._values["excess_pitch_deg"]
        )

    def _state(self) -> dict[str, float]:
        state = {
            "phase_index": float(self._phase_index),
            "progress": self._progress(),
            "safe": float(self._last_safe),
            "last_update": float(self._last_update),
            "last_rollback": float(self._last_rollback),
            "advance_count": float(self._advance_count),
            "rollback_count": float(self._rollback_count),
            "phase_update_count": float(self._phase_update_count),
            "phase_completed_episodes": float(self._phase_completed_episodes),
            "phase_advance_count": float(self._phase_advance_count),
            "phase_rollback_count": float(self._phase_rollback_count),
            "completed_episodes": float(self._total_completed),
            "window_episodes": float(len(self._success)),
            "success_rate": _mean(self._success),
            "target_reached_rate": _mean(self._target_reached),
            "reached_but_not_success_rate": _mean(
                [
                    max(0.0, target - success)
                    for target, success in zip(
                        self._target_reached,
                        self._success,
                        strict=False,
                    )
                ]
            ),
            "time_out_rate": _mean(self._time_out),
            "excess_pitch_rate": _mean(self._excess_pitch),
            "excess_depth_error_rate": _mean(self._excess_depth_error),
            "excess_xy_drift_rate": _mean(self._excess_xy_drift),
            "first_done_time_mean_s": _mean(self._first_done_s),
            "xy_peak_p95_m": _quantile(self._xy_peak_m, 0.95),
            "pitch_abs_peak_p95_deg": math.degrees(
                _quantile(self._pitch_peak_rad, 0.95)
            ),
            "depth_abs_error_mean_m": _mean(self._depth_abs_error_m),
            "depth_abs_error_p95_m": _quantile(self._depth_abs_error_m, 0.95),
            "pitch_abs_mean_rad": _mean(self._pitch_abs_rad),
            "pitch_abs_p95_rad": _quantile(self._pitch_abs_rad, 0.95),
            "yaw_abs_error_mean_rad": _mean(self._yaw_abs_error_rad),
            "yaw_abs_error_p95_rad": _quantile(self._yaw_abs_error_rad, 0.95),
            "root_ang_speed_mean_rad_s": _mean(self._root_ang_speed_rad_s),
            "root_ang_speed_p95_rad_s": _quantile(self._root_ang_speed_rad_s, 0.95),
            "action_l2_mean": _mean(self._action_l2),
            "action_l2_p95": _quantile(self._action_l2, 0.95),
            "saturation_time_mean": _mean(self._saturation),
            "saturation_time_p95": _quantile(self._saturation, 0.95),
            "k_xy": self._values["k_xy"],
            "k_pitch": self._values["k_pitch"],
            "k_yaw": self._values["k_yaw"],
            "k_depth": self._values["k_depth"],
            "k_smooth": self._values["k_smooth"],
            "k_action_effort": self._values["k_action_effort"],
            "k_thruster_saturation": self._values["k_thruster_saturation"],
            "thruster_saturation_threshold": self._values[
                "thruster_saturation_threshold"
            ],
            "settle_window_s": self._values["settle_window_s"],
            "settle_pitch_limit_deg": self._values["settle_pitch_limit_deg"],
            "settle_yaw_limit_deg": self._values["settle_yaw_limit_deg"],
            "settle_ang_vel_limit_rad_s": self._values[
                "settle_ang_vel_limit_rad_s"
            ],
            "settle_depth_error_limit_m": self._values[
                "settle_depth_error_limit_m"
            ],
            "settle_xy_drift_limit_m": self._values["settle_xy_drift_limit_m"],
            "excess_pitch_deg": self._values["excess_pitch_deg"],
        }
        state.update(self._settle_condition_rates())
        phase = self._PHASES[self._phase_index]
        state.update(
            {f"phase_is_{name}": float(name == phase) for name in self._PHASES}
        )
        state.update(self._gate_state())
        return state

    def _gate_state(self) -> dict[str, float]:
        phase = self._PHASES[self._phase_index]
        phase_index = self._phase_index
        has_window = bool(self._success)
        empty_metric_default = 0.0 if not has_window else float("inf")
        success_rate = _mean(self._success)
        xy_peak_p95 = _quantile(self._xy_peak_m, 0.95, default=empty_metric_default)
        pitch_peak_p95_deg = math.degrees(
            _quantile(self._pitch_peak_rad, 0.95, default=empty_metric_default)
        )
        depth_mean = _mean(self._depth_abs_error_m, default=empty_metric_default)
        root_ang_speed_mean = _mean(
            self._root_ang_speed_rad_s, default=empty_metric_default
        )

        success_limit = self._schedule.success_advance_threshold
        pitch_limit = self._values["excess_pitch_deg"] + 1.0
        xy_limit = self._xy_advance_limit_for_phase(phase)
        root_ang_speed_limit = self._values["settle_ang_vel_limit_rad_s"]
        depth_limit = self._values["settle_depth_error_limit_m"]
        settle_xy_limit = self._values["settle_xy_drift_limit_m"]
        xy_rollback_limit = self._xy_rollback_limit_for_phase(phase)
        pitch_rollback_limit = max(
            self._schedule.pitch_peak_rollback_max_deg,
            self._values["excess_pitch_deg"] + 2.0,
        )

        require_root_ang_speed = phase_index >= self._PHASES.index("settle_ang_vel")
        require_depth = phase_index >= self._PHASES.index("settle_depth")
        require_settle_xy = phase_index >= self._PHASES.index("settle_xy")

        success_margin = success_rate - success_limit
        xy_peak_margin_m = xy_limit - xy_peak_p95
        pitch_peak_margin_deg = pitch_limit - pitch_peak_p95_deg
        pitch_margin_to_excess_deg = (
            self._values["excess_pitch_deg"] - pitch_peak_p95_deg
        )
        root_ang_speed_margin_rad_s = root_ang_speed_limit - root_ang_speed_mean
        depth_margin_m = depth_limit - depth_mean
        settle_xy_margin_m = settle_xy_limit - xy_peak_p95
        success_rollback_margin = (
            success_rate - self._schedule.success_rollback_threshold
        )
        xy_rollback_margin_m = xy_rollback_limit - xy_peak_p95
        pitch_rollback_margin_deg = pitch_rollback_limit - pitch_peak_p95_deg
        saturation_mean = _mean(self._saturation, default=empty_metric_default)
        action_l2_mean = _mean(self._action_l2, default=empty_metric_default)
        saturation_margin = self._schedule.saturation_rollback_max - saturation_mean
        action_l2_margin = self._schedule.action_l2_rollback_max - action_l2_mean

        return {
            "advance_success_margin": success_margin,
            "advance_xy_peak_margin_m": xy_peak_margin_m,
            "advance_pitch_peak_margin_deg": pitch_peak_margin_deg,
            "pitch_margin_to_excess_deg": pitch_margin_to_excess_deg,
            "root_ang_speed_margin_rad_s": root_ang_speed_margin_rad_s,
            "depth_margin_m": depth_margin_m,
            "settle_xy_margin_m": settle_xy_margin_m,
            "advance_success_limit": success_limit,
            "advance_xy_peak_limit_m": xy_limit,
            "advance_pitch_peak_limit_deg": pitch_limit,
            "advance_root_ang_speed_limit_rad_s": root_ang_speed_limit,
            "advance_depth_limit_m": depth_limit,
            "advance_settle_xy_limit_m": settle_xy_limit,
            "rollback_xy_peak_limit_m": xy_rollback_limit,
            "rollback_pitch_peak_limit_deg": pitch_rollback_limit,
            "rollback_success_limit": self._schedule.success_rollback_threshold,
            "rollback_saturation_limit": self._schedule.saturation_rollback_max,
            "rollback_action_l2_limit": self._schedule.action_l2_rollback_max,
            "rollback_success_margin": success_rollback_margin,
            "rollback_xy_peak_margin_m": xy_rollback_margin_m,
            "rollback_pitch_peak_margin_deg": pitch_rollback_margin_deg,
            "rollback_saturation_margin": saturation_margin,
            "rollback_action_l2_margin": action_l2_margin,
            "advance_blocked_by_success": float(success_margin < 0.0),
            "advance_blocked_by_xy": float(xy_peak_margin_m < 0.0),
            "advance_blocked_by_pitch": float(pitch_peak_margin_deg < 0.0),
            "advance_blocked_by_root_ang_speed": float(
                require_root_ang_speed and root_ang_speed_margin_rad_s < 0.0
            ),
            "advance_blocked_by_depth": float(require_depth and depth_margin_m < 0.0),
            "advance_blocked_by_settle_xy": float(
                require_settle_xy and settle_xy_margin_m < 0.0
            ),
            "rollback_blocked_by_success": float(success_rollback_margin < 0.0),
            "rollback_blocked_by_xy": float(xy_rollback_margin_m < 0.0),
            "rollback_blocked_by_pitch": float(pitch_rollback_margin_deg < 0.0),
            "rollback_blocked_by_saturation": float(saturation_margin < 0.0),
            "rollback_blocked_by_action_l2": float(action_l2_margin < 0.0),
        }

    def _settle_condition_rates(self) -> dict[str, float]:
        settle_pitch_limit_rad = math.radians(self._values["settle_pitch_limit_deg"])
        settle_yaw_limit_rad = math.radians(self._values["settle_yaw_limit_deg"])
        settle_ang_vel_limit = self._values["settle_ang_vel_limit_rad_s"]
        settle_depth_limit = self._values["settle_depth_error_limit_m"]
        settle_xy_limit = self._values["settle_xy_drift_limit_m"]

        pitch_ok = [float(value <= settle_pitch_limit_rad) for value in self._pitch_abs_rad]
        yaw_ok = [float(value <= settle_yaw_limit_rad) for value in self._yaw_abs_error_rad]
        ang_vel_ok = [
            float(value <= settle_ang_vel_limit) for value in self._root_ang_speed_rad_s
        ]
        depth_ok = [float(value <= settle_depth_limit) for value in self._depth_abs_error_m]
        xy_ok = [float(value <= settle_xy_limit) for value in self._xy_peak_m]
        all_ok = [
            float(all(values))
            for values in zip(pitch_ok, yaw_ok, ang_vel_ok, depth_ok, xy_ok, strict=False)
        ]
        return {
            "settle_pitch_ok_rate": _mean(pitch_ok),
            "settle_yaw_ok_rate": _mean(yaw_ok),
            "settle_ang_vel_ok_rate": _mean(ang_vel_ok),
            "settle_depth_ok_rate": _mean(depth_ok),
            "settle_xy_ok_rate": _mean(xy_ok),
            "settle_all_ok_rate": _mean(all_ok),
        }

    def _xy_advance_limit_for_phase(self, phase: str) -> float:
        if phase in {
            "settle_xy",
            "smoothness",
            "saturation_weight",
            "saturation_threshold",
            "done",
        }:
            return max(
                self._schedule.xy_peak_advance_max_m,
                self._values["settle_xy_drift_limit_m"],
            )
        return max(
            self._schedule.xy_peak_advance_max_m,
            min(
                self._schedule.xy_peak_rollback_max_m,
                self._values["settle_xy_drift_limit_m"],
            ),
        )

    def _xy_rollback_limit_for_phase(self, phase: str) -> float:
        if self._PHASES.index(phase) >= self._PHASES.index("settle_xy"):
            return max(
                self._schedule.xy_peak_rollback_max_m,
                self._values["settle_xy_drift_limit_m"],
            )
        return self._schedule.xy_peak_rollback_max_m

    def _progress(self) -> float:
        fields = [
            field
            for phase in self._PHASE_FIELDS
            for field in self._PHASE_FIELDS[phase]
        ]
        progress = []
        for field in fields:
            start = self._start[field]
            goal = self._goal[field]
            current = self._values[field]
            if math.isclose(start, goal, rel_tol=0.0, abs_tol=1.0e-12):
                progress.append(1.0)
            else:
                progress.append(abs(current - start) / abs(goal - start))
        return max(0.0, min(1.0, _mean(progress)))


def build_post_c3l_polish_curriculum(
    *,
    start_stage: RollCurriculumStage,
    goal_stage: RollCurriculumStage,
    term_name: str = POST_C3L_POLISH_AUTO_CURRICULUM,
    **schedule_overrides: Any,
) -> dict[str, CurriculumTermCfg]:
    schedule = PostC3LPolishSchedule(
        start_stage=start_stage,
        goal_stage=goal_stage,
        **schedule_overrides,
    )
    return {
        term_name: CurriculumTermCfg(
            func=PostC3LPolishCurriculum,
            params={"schedule": schedule},
        )
    }


def _stage_values(stage: RollCurriculumStage) -> dict[str, float]:
    if stage.settle_xy_drift_limit_m is None:
        raise ValueError("post-c3l polish requires settle_xy_drift_limit_m.")
    return {
        "target_roll_deg": float(stage.target_roll_deg),
        "k_xy": float(stage.k_xy),
        "k_pitch": float(stage.k_pitch),
        "k_yaw": float(stage.k_yaw),
        "k_depth": float(stage.k_depth),
        "k_smooth": float(stage.k_smooth),
        "k_action_effort": float(stage.k_action_effort),
        "k_thruster_saturation": float(stage.k_thruster_saturation),
        "thruster_saturation_threshold": float(stage.thruster_saturation_threshold),
        "settle_window_s": float(stage.settle_window_s),
        "settle_pitch_limit_deg": float(stage.settle_pitch_limit_deg),
        "settle_yaw_limit_deg": float(stage.settle_yaw_limit_deg),
        "settle_ang_vel_limit_rad_s": float(stage.settle_ang_vel_limit_rad_s),
        "settle_depth_error_limit_m": float(stage.settle_depth_error_limit_m),
        "settle_xy_drift_limit_m": float(stage.settle_xy_drift_limit_m),
        "excess_pitch_deg": float(stage.excess_pitch_deg),
    }


def _env_id_tensor(env: ManagerBasedRlEnv, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None or isinstance(env_ids, slice):
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return env_ids.to(device=env.device, dtype=torch.long)


def _episode_metric_values(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    name: str,
) -> torch.Tensor | None:
    metrics_manager = env.metrics_manager
    active_terms = getattr(metrics_manager, "active_terms", [])
    if name not in active_terms:
        return None
    index = active_terms.index(name)
    term_cfgs = getattr(metrics_manager, "_term_cfgs", None)
    step_values = getattr(metrics_manager, "_step_values", None)
    episode_sums = getattr(metrics_manager, "_episode_sums", None)
    step_count = getattr(metrics_manager, "_step_count", None)
    if term_cfgs is None or step_values is None or episode_sums is None or step_count is None:
        return None
    if term_cfgs[index].reduce == "last":
        return step_values[env_ids, index]
    return episode_sums[name][env_ids] / torch.clamp(step_count[env_ids].float(), min=1.0)


def _move_toward(current: float, target: float, step: float) -> float:
    if math.isclose(current, target, rel_tol=0.0, abs_tol=1.0e-12):
        return target
    if current < target:
        return min(current + abs(step), target)
    return max(current - abs(step), target)


def _mean(values: deque[float] | list[float], default: float = 0.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def _quantile(
    values: deque[float] | list[float],
    q: float,
    default: float = 0.0,
) -> float:
    if not values:
        return default
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[index]


__all__ = [
    "POST_C3L_POLISH_AUTO_CURRICULUM",
    "PostC3LPolishCurriculum",
    "PostC3LPolishSchedule",
    "build_post_c3l_polish_curriculum",
]
