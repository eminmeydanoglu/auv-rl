"""Phase presets for training the Taluy roll expert policy."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RollCurriculumStage:
    """Static curriculum phase applied when building a roll environment."""

    name: str
    description: str
    target_roll_deg: float
    episode_length_s: float
    settle_window_s: float
    k_prog: float
    k_xy: float
    k_pitch: float
    k_yaw: float
    k_depth: float
    k_smooth: float
    excess_pitch_deg: float
    excess_depth_error_m: float
    excess_xy_drift_m: float
    settle_pitch_limit_deg: float
    settle_yaw_limit_deg: float
    settle_ang_vel_limit_rad_s: float
    settle_depth_error_limit_m: float
    terminal_success_weight: float = 100.0
    terminal_failure_weight: float = -50.0

    def roll_env_kwargs(self) -> dict[str, float]:
        """Return keyword arguments consumed by ``make_roll_env_cfg``."""
        return {
            "target_roll_deg": self.target_roll_deg,
            "settle_window_s": self.settle_window_s,
            "k_prog": self.k_prog,
            "k_xy": self.k_xy,
            "k_pitch": self.k_pitch,
            "k_yaw": self.k_yaw,
            "k_depth": self.k_depth,
            "k_smooth": self.k_smooth,
            "excess_pitch_deg": self.excess_pitch_deg,
            "excess_depth_error_m": self.excess_depth_error_m,
            "excess_xy_drift_m": self.excess_xy_drift_m,
            "settle_pitch_limit_deg": self.settle_pitch_limit_deg,
            "settle_yaw_limit_deg": self.settle_yaw_limit_deg,
            "settle_ang_vel_limit_rad_s": self.settle_ang_vel_limit_rad_s,
            "settle_depth_error_limit_m": self.settle_depth_error_limit_m,
            "terminal_success_weight": self.terminal_success_weight,
            "terminal_failure_weight": self.terminal_failure_weight,
        }


ROLL_CURRICULUM_STAGES: dict[str, RollCurriculumStage] = {
    "c0_90_discovery": RollCurriculumStage(
        name="c0_90_discovery",
        description="Discover that positive body roll torque increases phi_total.",
        target_roll_deg=90.0,
        episode_length_s=8.0,
        settle_window_s=0.1,
        k_prog=10.0,
        k_xy=0.0,
        k_pitch=0.2,
        k_yaw=0.05,
        k_depth=0.1,
        k_smooth=0.001,
        excess_pitch_deg=89.0,
        excess_depth_error_m=2.0,
        excess_xy_drift_m=10.0,
        settle_pitch_limit_deg=45.0,
        settle_yaw_limit_deg=90.0,
        settle_ang_vel_limit_rad_s=4.0,
        settle_depth_error_limit_m=2.0,
        terminal_success_weight=60.0,
        terminal_failure_weight=-20.0,
    ),
    "c1_180_inversion": RollCurriculumStage(
        name="c1_180_inversion",
        description="Reach inverted attitude while keeping catastrophic errors bounded.",
        target_roll_deg=180.0,
        episode_length_s=10.0,
        settle_window_s=0.1,
        k_prog=8.0,
        k_xy=0.0,
        k_pitch=0.4,
        k_yaw=0.1,
        k_depth=0.4,
        k_smooth=0.003,
        excess_pitch_deg=89.0,
        excess_depth_error_m=1.8,
        excess_xy_drift_m=10.0,
        settle_pitch_limit_deg=35.0,
        settle_yaw_limit_deg=75.0,
        settle_ang_vel_limit_rad_s=3.0,
        settle_depth_error_limit_m=1.5,
        terminal_success_weight=80.0,
        terminal_failure_weight=-30.0,
    ),
    "c2_360_full_turn": RollCurriculumStage(
        name="c2_360_full_turn",
        description="Complete one full roll with moderate attitude/depth discipline.",
        target_roll_deg=360.0,
        episode_length_s=14.0,
        settle_window_s=0.2,
        k_prog=6.0,
        k_xy=0.0,
        k_pitch=0.6,
        k_yaw=0.2,
        k_depth=0.6,
        k_smooth=0.005,
        excess_pitch_deg=85.0,
        excess_depth_error_m=1.5,
        excess_xy_drift_m=10.0,
        settle_pitch_limit_deg=25.0,
        settle_yaw_limit_deg=50.0,
        settle_ang_vel_limit_rad_s=2.0,
        settle_depth_error_limit_m=1.0,
        terminal_success_weight=100.0,
        terminal_failure_weight=-40.0,
    ),
    "c3_720_no_hard_stop": RollCurriculumStage(
        name="c3_720_no_hard_stop",
        description="Complete two rolls without requiring a precise final stop.",
        target_roll_deg=720.0,
        episode_length_s=20.0,
        settle_window_s=0.2,
        k_prog=5.0,
        k_xy=0.0,
        k_pitch=0.8,
        k_yaw=0.3,
        k_depth=0.8,
        k_smooth=0.008,
        excess_pitch_deg=82.0,
        excess_depth_error_m=1.25,
        excess_xy_drift_m=10.0,
        settle_pitch_limit_deg=18.0,
        settle_yaw_limit_deg=30.0,
        settle_ang_vel_limit_rad_s=1.5,
        settle_depth_error_limit_m=0.6,
        terminal_success_weight=100.0,
        terminal_failure_weight=-50.0,
    ),
    "c4_720_settle": RollCurriculumStage(
        name="c4_720_settle",
        description="Complete two rolls and settle for nominal deployment quality.",
        target_roll_deg=720.0,
        episode_length_s=20.0,
        settle_window_s=1.0,
        k_prog=4.0,
        k_xy=0.05,
        k_pitch=1.0,
        k_yaw=0.5,
        k_depth=1.0,
        k_smooth=0.01,
        excess_pitch_deg=80.0,
        excess_depth_error_m=1.0,
        excess_xy_drift_m=10.0,
        settle_pitch_limit_deg=10.0,
        settle_yaw_limit_deg=15.0,
        settle_ang_vel_limit_rad_s=0.25,
        settle_depth_error_limit_m=0.15,
        terminal_success_weight=100.0,
        terminal_failure_weight=-50.0,
    ),
}


def get_roll_curriculum_stage(name: str) -> RollCurriculumStage:
    """Return a named roll curriculum stage or raise a helpful error."""
    try:
        return ROLL_CURRICULUM_STAGES[name]
    except KeyError as exc:
        names = ", ".join(sorted(ROLL_CURRICULUM_STAGES))
        raise ValueError(f"Unknown roll curriculum stage '{name}'. Choices: {names}.") from exc


__all__ = [
    "ROLL_CURRICULUM_STAGES",
    "RollCurriculumStage",
    "get_roll_curriculum_stage",
]
