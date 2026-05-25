#!/usr/bin/env bash
set -euo pipefail

AUV_HOME="${AUV_HOME:-$HOME/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

cd "$AUV_PROJECT_DIR"
mkdir -p logs/remote_runs "$AUV_HOME/logs/slurm"

BASE_CKPT="${BASE_CKPT:-logs/rsl_rl/taluy_roll_v1/2026-05-23_19-55-21_c3l_720_xy_guard_from_c3i_sat010_e699_a100x4_20260523_183212/model_699.pt}"
BASE_LABEL="${BASE_LABEL:-c3l_e699}"

if [ ! -f "$BASE_CKPT" ]; then
  echo "[submit-c3l-followup] Missing base checkpoint: $BASE_CKPT" >&2
  echo "[submit-c3l-followup] Set BASE_CKPT=... to override or rsync c3l to the cluster first." >&2
  exit 1
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
SUBMISSIONS="logs/remote_runs/c3l_followup_submissions_${STAMP}.tsv"

COMMON_EXPORTS=(
  "AUV_HOME=$AUV_HOME"
  "AUV_PROJECT_DIR=$AUV_PROJECT_DIR"
  "RESUME_CHECKPOINT=$BASE_CKPT"
  "RESUME_MODE=weights-only"
  "NUM_NODES=${NUM_NODES:-4}"
  "NUM_GPUS_PER_NODE=${NUM_GPUS_PER_NODE:-1}"
  "NUM_ENVS_PER_GPU=${NUM_ENVS_PER_GPU:-1024}"
  "NUM_STEPS_PER_ENV=${NUM_STEPS_PER_ENV:-256}"
  "ITERS=${ITERS:-700}"
  "SAVE_INTERVAL=${SAVE_INTERVAL:-50}"
  "LEARNING_RATE=${LEARNING_RATE:-3e-4}"
  "ENTROPY_COEF=${ENTROPY_COEF:-0.003}"
  "DESIRED_KL=${DESIRED_KL:-0.006}"
)

submit_stage() {
  local job_name="$1"
  local stage="$2"
  local run_name="${stage}_from_${BASE_LABEL}_a100x4_${STAMP}"
  local log_path="logs/remote_runs/${run_name}.log"
  local exports
  local output
  local job_id

  exports="$(IFS=,; echo "ALL,${COMMON_EXPORTS[*]},STAGE=${stage},RUN_NAME=${run_name},RUN_LOG_PATH=${log_path}")"
  printf '[submit-c3l-followup] stage=%s run_name=%s\n' "$stage" "$run_name"
  output="$(
    sbatch \
      --job-name "$job_name" \
      --output "$AUV_HOME/logs/slurm/%x-%j.out" \
      --error "$AUV_HOME/logs/slurm/%x-%j.err" \
      --export "$exports" \
      jobs/slurm/taluy_roll_a100x4.sbatch
  )"
  job_id="$(grep -Eo '[0-9]+' <<<"$output" | tail -n 1)"
  printf '[submit-c3l-followup] %s\n' "$output"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$job_id" "$job_name" "$stage" "$run_name" "$BASE_CKPT" "$log_path" \
    | tee -a "$SUBMISSIONS"
}

printf 'job_id\tjob_name\tstage\trun_name\tresume_checkpoint\tlog_path\n' > "$SUBMISSIONS"
printf '[submit-c3l-followup] base_ckpt=%s\n' "$BASE_CKPT"
submit_stage "roll-c3n-sat-soft" "c3n_720_c3l_sat_soft"
submit_stage "roll-c3o-sat-thr" "c3o_720_c3l_sat_threshold"
submit_stage "roll-c3p-deploy" "c3p_720_c3l_deploy_polish"
printf '[submit-c3l-followup] wrote=%s\n' "$SUBMISSIONS"
