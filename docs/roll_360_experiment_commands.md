# Roll 360 Experiment Commands

This runbook starts all 360-degree experiments from the clean C1 baseline:

```bash
logs/rsl_rl/taluy_roll_v1/2026-04-26_22-05-07_c1_180_from_c0_baseline/model_360.pt
```

All runs should use `--resume-mode weights-only`.

## Setup Check

Run once on the remote machine:

```bash
cd /teamspace/studios/this_studio/auvrl
export PATH="$HOME/.local/bin:$PATH"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/teamspace/studios/this_studio/.uv-cache}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/teamspace/studios/this_studio/.matplotlib}"
export MUJOCO_GL=egl

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" logs/remote_runs

uv sync --python 3.12
uv run --with pytest pytest tests/tasks/roll
```

## Common Variables

```bash
cd /teamspace/studios/this_studio/auvrl

export SOURCE_CKPT="logs/rsl_rl/taluy_roll_v1/2026-04-26_22-05-07_c1_180_from_c0_baseline/model_360.pt"
export NUM_ENVS=1024
export NUM_STEPS_PER_ENV=256
export ITERS=700
export SAVE_INTERVAL=50
export LEARNING_RATE=3e-4
export ENTROPY_COEF=0.003
export DESIRED_KL=0.006
```

Helper function for tmux runs:

```bash
start_roll360_stage() {
  local stage="$1"
  local session="$2"
  local suffix="$3"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local run_name="${stage}_from_ckpt_env${NUM_ENVS}_steps${NUM_STEPS_PER_ENV}_${suffix}_${stamp}"
  local log_path="logs/remote_runs/${run_name}.log"

  tmux new -d -s "$session" "
    cd /teamspace/studios/this_studio/auvrl &&
    export PATH=\"\$HOME/.local/bin:\$PATH\" &&
    export UV_CACHE_DIR=\"${UV_CACHE_DIR}\" &&
    export MPLCONFIGDIR=\"${MPLCONFIGDIR}\" &&
    export MUJOCO_GL=egl &&
    mkdir -p logs/remote_runs &&
    uv run python -u -m auvrl.scripts.train.taluy_roll \
      --device cuda \
      --num-envs ${NUM_ENVS} \
      --num-steps-per-env ${NUM_STEPS_PER_ENV} \
      --curriculum-stage ${stage} \
      --resume-checkpoint ${SOURCE_CKPT} \
      --resume-mode weights-only \
      --iterations ${ITERS} \
      --save-interval ${SAVE_INTERVAL} \
      --learning-rate ${LEARNING_RATE} \
      --entropy-coef ${ENTROPY_COEF} \
      --desired-kl ${DESIRED_KL} \
      --run-name ${run_name} \
      2>&1 | tee ${log_path}
  "

  echo "Started ${session}"
  echo "Run name: ${run_name}"
  echo "Log: ${log_path}"
}
```

## Probe Runs

Baseline after the reward-normalization and short-settle patch:

```bash
start_roll360_stage c2a_360_reach roll-c2a-baseline baseline
```

Primary candidate, light attitude discipline and C1-level per-degree progress:

```bash
start_roll360_stage c2r1_360_reach_light roll-c2r1 r1
```

If only one run can be started first, start `c2r1_360_reach_light`.

## Manual Chain

After `c2r1_360_reach_light`, pick the best checkpoint by deterministic eval, not by the final checkpoint alone.

Set the selected checkpoint:

```bash
export SOURCE_CKPT="logs/rsl_rl/taluy_roll_v1/<c2r1_run_dir>/model_<best>.pt"
```

Then start the short-settle stage:

```bash
start_roll360_stage c2r2_360_short_settle roll-c2r2 r2_from_best_r1
```

After `c2r2_360_short_settle`, select its best checkpoint and continue:

```bash
export SOURCE_CKPT="logs/rsl_rl/taluy_roll_v1/<c2r2_run_dir>/model_<best>.pt"

start_roll360_stage c2r3_360_stable_settle roll-c2r3 r3_from_best_r2
```

## Attach And Logs

```bash
tmux ls
tmux attach -t roll-c2r1
tail -f logs/remote_runs/*.log
```

Latest run directories:

```bash
find logs/rsl_rl/taluy_roll_v1 -maxdepth 1 -type d -printf "%T@ %p\n" \
  | sort -n \
  | tail -20
```

Latest checkpoints in a run:

```bash
find logs/rsl_rl/taluy_roll_v1/<run_dir> -maxdepth 1 -type f -name "model_*.pt" \
  -printf "%f %p\n" \
  | sort -V \
  | tail -20
```

## Decision Rules

Prefer a checkpoint if:

- deterministic `task_success` is high,
- `target_reached_last` is stable,
- `roll_progress_ratio_last` is near or above `1.0`,
- `xy_drift_m`, `depth_abs_error_m`, `pitch_abs_rad`, and `yaw_abs_error_rad` do not collapse,
- `root_ang_speed_rad_s` is not worsening across the selected window.

Stop early if by roughly 300-400 iterations:

- `target_reached_last` remains low,
- `roll_progress_ratio_last` is stuck around partial-roll behavior,
- `task_success` stays zero even while `Train/mean_reward` rises.
