#!/usr/bin/env bash
# Submit c3j_720_polish training on UHeM Altay A100x4, warm-starting from sat010_e699.
set -euo pipefail

AUV_HOME="${AUV_HOME:-$HOME/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

cd "$AUV_PROJECT_DIR"
mkdir -p logs/remote_runs "$AUV_HOME/logs/slurm"

# sat010 checkpoint (final epoch 699) on UHeM. Override via env if the path
# differs on the cluster.
BASE_CKPT="${BASE_CKPT:-logs/rsl_rl/taluy_roll_v1/2026-05-20_14-57-16_c3i_720_hold_0p10_sat010_from_c3h500_env1024_steps256_sat010_20260520_145256/model_699.pt}"

if [ ! -f "$BASE_CKPT" ]; then
  echo "[submit-c3j] Missing base checkpoint: $BASE_CKPT" >&2
  echo "[submit-c3j] Set BASE_CKPT=... to override or rsync sat010 to the cluster first." >&2
  exit 1
fi

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_NAME="${RUN_NAME:-c3j_720_polish_from_sat010_a100x4_${STAMP}}"
JOB_NAME="${JOB_NAME:-roll-c3j-polish}"
LOG_PATH="logs/remote_runs/${RUN_NAME}.log"

exports=(
  "AUV_HOME=$AUV_HOME"
  "AUV_PROJECT_DIR=$AUV_PROJECT_DIR"
  "STAGE=c3j_720_polish"
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
  "RUN_NAME=$RUN_NAME"
  "RUN_LOG_PATH=$LOG_PATH"
)

export_str="$(IFS=,; echo "ALL,${exports[*]}")"

printf '[submit-c3j] run_name=%s\n' "$RUN_NAME"
printf '[submit-c3j] base_ckpt=%s\n' "$BASE_CKPT"
printf '[submit-c3j] export=%s\n' "$export_str"

output="$(
  sbatch \
    --job-name "$JOB_NAME" \
    --output "$AUV_HOME/logs/slurm/%x-%j.out" \
    --error "$AUV_HOME/logs/slurm/%x-%j.err" \
    --export "$export_str" \
    jobs/slurm/taluy_roll_a100x4.sbatch
)"

printf '[submit-c3j] %s\n' "$output"
job_id="$(grep -Eo '[0-9]+' <<<"$output" | tail -n 1)"
printf '[submit-c3j] job_id=%s\n' "$job_id"
printf '[submit-c3j] tail logs: tail -F %s/logs/slurm/%s-%s.out\n' "$AUV_HOME" "$JOB_NAME" "$job_id"
printf '[submit-c3j] training stdout: tail -F %s/%s\n' "$AUV_PROJECT_DIR" "$LOG_PATH"
