#!/usr/bin/env bash
set -euo pipefail

AUV_HOME="${AUV_HOME:-$HOME/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

cd "$AUV_PROJECT_DIR"
mkdir -p logs/remote_runs "$AUV_HOME/logs/slurm"

BASE_CKPT="${BASE_CKPT:-logs/rsl_rl/taluy_roll_v1/2026-05-04_17-09-30_c2c_360_hold_0p10_from_c2b_best_env1024_steps256_20260504_170830/model_650.pt}"
BASE_LABEL="${BASE_LABEL:-b_from_a650}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUBMISSIONS="logs/remote_runs/roll_720_wave_submissions_${STAMP}.tsv"

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
  local run_name="${stage}_from_${BASE_LABEL}_env1024_steps256_${suffix}_${STAMP}"
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

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$job_name" "$stage" "$run_name" "$BASE_CKPT" "$log_path" \
    | tee -a "$SUBMISSIONS"
}

printf 'job_id\tjob_name\tstage\trun_name\tresume_checkpoint\tlog_path\n' > "$SUBMISSIONS"
submit_stage "roll-720-loose" "c3c_720_reach_loose_control" "loose"
submit_stage "roll-720-xy" "c3d_720_reach_xy_moderate" "xy_moderate"
submit_stage "roll-720-hold05" "c3e_720_hold_0p05_xy_light" "hold05"
submit_stage "roll-720-hold10" "c3f_720_hold_0p10_soft" "hold10"

printf 'WROTE %s\n' "$SUBMISSIONS"
