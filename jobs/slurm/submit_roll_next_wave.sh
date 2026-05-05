#!/usr/bin/env bash
set -euo pipefail

AUV_HOME="${AUV_HOME:-$HOME/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

cd "$AUV_PROJECT_DIR"
mkdir -p logs/remote_runs "$AUV_HOME/logs/slurm"

BASE_CKPT="${BASE_CKPT:-logs/rsl_rl/taluy_roll_v1/2026-05-03_22-13-58_c2a_360_reach_from_c1_baseline_env1024_steps256_c2a_20260503_221222/model_699.pt}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUBMISSIONS="logs/remote_runs/roll_next_wave_submissions_${STAMP}.tsv"

COMMON_EXPORTS=(
  "AUV_HOME=$AUV_HOME"
  "AUV_PROJECT_DIR=$AUV_PROJECT_DIR"
  "RESUME_CHECKPOINT=$BASE_CKPT"
  "RESUME_MODE=weights-only"
  "NUM_ENVS=1024"
  "NUM_STEPS_PER_ENV=256"
  "ITERS=700"
  "SAVE_INTERVAL=50"
  "LEARNING_RATE=3e-4"
  "ENTROPY_COEF=0.003"
  "DESIRED_KL=0.006"
)

submit_stage() {
  local job_name="$1"
  local stage="$2"
  local suffix="$3"
  local run_name="${stage}_from_c2a699_env1024_steps256_${suffix}_${STAMP}"
  local log_path="logs/remote_runs/${run_name}.log"
  local exports

  exports="$(IFS=,; echo "${COMMON_EXPORTS[*]},STAGE=${stage},RUN_NAME=${run_name},RUN_LOG_PATH=${log_path}")"
  local output
  output="$(
    sbatch \
      --job-name "$job_name" \
      --output "$AUV_HOME/logs/slurm/%x-%j.out" \
      --error "$AUV_HOME/logs/slurm/%x-%j.err" \
      --export "ALL,${exports}" \
      jobs/slurm/taluy_roll.sbatch
  )"
  local job_id
  job_id="$(grep -Eo '[0-9]+' <<<"$output" | tail -n 1)"

  printf '%s\t%s\t%s\t%s\t%s\n' "$job_id" "$job_name" "$stage" "$run_name" "$log_path" \
    | tee -a "$SUBMISSIONS"
}

printf 'job_id\tjob_name\tstage\trun_name\tlog_path\n' > "$SUBMISSIONS"
submit_stage "roll-A-c2b" "c2b_360_hold_0p05" "A"
submit_stage "roll-B-direct" "c2c_360_hold_0p10" "B_direct"
submit_stage "roll-C-540" "c3a_540_reach_0p05" "C"
submit_stage "roll-D-720" "c3b_720_reach_0p02" "D"

printf 'WROTE %s\n' "$SUBMISSIONS"
printf '\nDeferred second wave after A eval:\n'
printf '  source = best checkpoint from c2b_360_hold_0p05\n'
printf '  stage  = c2c_360_hold_0p10\n'
printf '  run    = c2c_360_hold_0p10_from_c2b_best\n'
