# Post-C3L Taluy Roll Auto-Curriculum Recommendations

Date: 2026-05-26

## Context

The 1000-iteration post-C3L auto-curriculum run completed successfully as a Slurm job, but it did not reach the strict C3Q target.

Run:

```text
post_c3l_polish_c3q_full1000_from_c3l_e699_a100x4q_20260526_023837
```

Key final training state:

```text
phase_index = 4
phase = settle_motion
progress = 0.5229
safe = 0
advance_count = 128
rollback_count = 12
success_rate = 0.998
first_done_time_mean_s = 10.2772
xy_peak_p95_m = 0.372
```

Best strict-C3Q eval-by-XY checkpoint:

```text
model_599
xy_peak_mean_m = 0.2413
first_done_mean_s = 11.5204
saturation_time_mean = 0.4897
success_rate = 0.0
task_success_rate = 0.0
excess_pitch_rate = 1.0
```

Final checkpoint:

```text
model_999
xy_peak_mean_m = 0.2982
first_done_mean_s = 12.1061
saturation_time_mean = 0.4914
success_rate = 0.0
task_success_rate = 0.0
excess_pitch_rate = 1.0
```

## Updated diagnosis

The strict-C3Q eval failure is real:

```text
success_rate = 0.0
excess_pitch_rate = 1.0
```

The earlier `eval_summary.json` terminal pitch values were misleading because terminal metrics were sampled after reset on the done step. The policy was actually hitting the hard pitch envelope immediately before termination.

Observed pre-done pitch behavior:

```text
model_49:  mean pitch ≈70.1 deg, p90 ≈73.9 deg, max ≈75.0 deg
model_99:  mean pitch ≈74.7 deg, p90 ≈74.8 deg, max ≈75.0 deg
model_599: mean pitch ≈74.8 deg, p90 ≈74.8 deg, max ≈75.0 deg
model_999: mean pitch ≈74.6 deg, p90 ≈74.7 deg, max ≈75.0 deg
```

Main mismatch:

```text
training start hard pitch limit: c3l excess_pitch_deg = 80
strict goal/eval hard pitch limit: c3q excess_pitch_deg = 75
```

The previous auto-curriculum mutated `task_success` settle parameters but did not mutate the hard `excess_pitch` termination. As a result, training allowed the policy to use the 75-degree pitch envelope, while goal eval treated it as catastrophic failure.

## Review of the current fixes

### Good fixes

- `eval_summary` now uses pre-done rows for terminal metrics via `first_done_step - 1`, which addresses the reset artifact.
- `pitch_abs_peak_rad` is now included as a terminal metric and as an environment metric.
- Auto-curriculum now tracks `pitch_abs_peak_rad` and logs `pitch_abs_peak_p95_deg`.
- Auto-curriculum now gates advance/rollback on pitch peak p95.
- Auto-curriculum now mutates `excess_pitch_deg` through the `excess_pitch` termination config.
- In-training eval now separates:
  - current-rule eval: captured live training reward/termination rules.
  - goal-rule eval: static target stage.
- `eval_rules.py` provides a useful mechanism for replaying the current live auto-curriculum rule set in evaluation.

### Important risks to watch

#### 1. C3Q remains too far as a one-shot target

Even after fixing pitch handling, the run stalled in `settle_motion` before reaching smoothness or saturation phases.

Current final state was still far from C3Q:

```text
settle_ang_vel_limit_rad_s = 1.85   target = 0.2618
settle_depth_error_limit_m = 1.35   target = 0.30
settle_xy_drift_limit_m = 0.84      target = 0.35
```

Recommendation: do not repeat the same C3L -> C3Q 1000-iteration run unchanged.

#### 2. `settle_motion` is too broad

Current `settle_motion` changes multiple difficult constraints together:

```text
settle_ang_vel_limit_rad_s
settle_depth_error_limit_m
settle_xy_drift_limit_m
excess_pitch_deg
```

This makes it hard to know which constraint blocks progression and creates coupled regressions.

Recommendation: split into smaller phases.

#### 3. XY advance threshold is slightly too strict for this phase

The final blocker was likely:

```text
xy_peak_p95_m = 0.372
xy_peak_advance_max_m = 0.35
```

This is a small miss, but it fully stopped progression.

Recommendation: allow a slightly softer advance gate during intermediate deploy-settle phases, while keeping rollback strict.

#### 4. Saturation should not be optimized yet

Saturation improved only slightly:

```text
c3l_e699 saturation_time_mean ≈ 0.505
auto model_599 saturation_time_mean ≈ 0.4897
auto model_999 saturation_time_mean ≈ 0.4914
```

This is not the main blocker. Early saturation pressure can destabilize XY/settle behavior.

Recommendation: push saturation to a later dedicated polish run.

## Recommended next target

Do not use strict C3Q as the immediate auto-curriculum goal.

Introduce an intermediate target, for example:

```text
c3r_720_c3l_deploy_settle
```

Suggested target values:

```text
target_roll_deg = 720
settle_window_s = 0.50
settle_pitch_limit_deg = 15
settle_yaw_limit_deg = 15
excess_pitch_deg = 75
settle_ang_vel_limit_rad_s = 1.20
settle_depth_error_limit_m = 0.90
settle_xy_drift_limit_m = 0.60
```

Reward weights should stay close to C3L/C3P-style deploy polish:

```text
k_xy = 0.30
k_pitch = 1.25
k_yaw = 0.50
k_depth = 0.65
k_smooth = 0.012 to 0.016
k_action_effort = 0.003 to 0.0045
k_thruster_saturation = 0.10 to 0.15 initially
thruster_saturation_threshold = 0.85 initially
```

## Recommended phase split

Replace the current broad `settle_motion` phase with smaller phases:

```text
observe
attitude_depth
settle_attitude
settle_window
hard_pitch_envelope
settle_ang_vel
settle_depth
settle_xy
smoothness
saturation_weight
saturation_threshold
done
```

Priority order:

1. Preserve task success.
2. Preserve XY and first-done timing.
3. Reduce hard pitch envelope from 80 deg to 75 deg.
4. Reduce settle angular velocity.
5. Reduce depth error.
6. Reduce settle XY limit.
7. Add smoothness.
8. Add saturation pressure only after the policy is stable.

## Recommended gate changes

For the next intermediate run:

```text
success_advance_threshold = 0.98
success_rollback_threshold = 0.95
xy_peak_advance_max_m = 0.40
xy_peak_rollback_max_m = 0.55 or 0.60
pitch_peak_advance_max_deg = 72.0
pitch_peak_rollback_max_deg = 78.0
first_done_advance_max_s = 10.8
first_done_rollback_max_s = 11.5
```

Rationale:

- `xy_peak_p95 = 0.372` should not completely block intermediate progression.
- The new pitch gate catches the real failure mode.
- Rollback remains strict enough to reject large regressions.

## Recommended run plan

### Step 1: short smoke

Use C3L or verified best checkpoint as the source.

Preferred source order:

1. `c3l_e699` for conservative stability.
2. `model_599` only after the fixed eval confirms it is not pitch-invalid under current rules.

Smoke run:

```text
iterations = 300
eval_interval = 50
save_interval = 10
goal = c3r_720_c3l_deploy_settle
```

Eval every checkpoint in two modes:

```text
current: captured live auto-curriculum rules
goal: static c3r target stage
```

Pass criteria:

```text
current success_rate >= 0.98
goal excess_pitch_rate near 0
pitch_abs_peak_p95_deg < 72
xy_peak_p95_m < 0.40
no repeated rollback loop
```

### Step 2: medium run

If smoke passes:

```text
iterations = 700 to 1000
goal = c3r_720_c3l_deploy_settle
```

Stop/inspect if:

```text
rollback_count grows continuously
pitch_peak_p95 approaches 78 deg
xy_peak_p95 stays above 0.45 for long windows
first_done_mean drifts above 11.5 s
```

### Step 3: strict C3Q only after C3R is stable

Run C3Q as a second-stage polish:

```text
source = best c3r checkpoint
goal = c3q_720_c3l_strict_settle
```

At that point strict values such as:

```text
settle_ang_vel_limit_rad_s = 0.2618
settle_depth_error_limit_m = 0.30
settle_xy_drift_limit_m = 0.35
```

can be approached without combining all difficulty into one jump from C3L.

## Final recommendation

The immediate next move should be:

1. Keep the pitch/eval fixes.
2. Add an intermediate `c3r_720_c3l_deploy_settle` target.
3. Split `settle_motion` into smaller phases.
4. Slightly relax intermediate XY advance gating to `0.40`.
5. Keep pitch peak p95 as a hard advance/rollback safety metric.
6. Defer serious saturation optimization to a later dedicated run.
7. Run a 300-iteration smoke before another full 1000-iteration job.

Short version:

```text
Do not repeat C3L -> C3Q directly.
Do C3L -> C3R first, with pitch-aware gates and current-vs-goal eval.
Then do C3R -> C3Q as the strict final polish.
```
