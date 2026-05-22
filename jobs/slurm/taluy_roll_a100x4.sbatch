#!/usr/bin/env bash
#SBATCH -J taluy-roll-mp
#SBATCH -A kds3by
#SBATCH -p a100q
#SBATCH --gres=gpu:a100:4
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 32
#SBATCH --time=12:00:00
#SBATCH --output=/ari/users/btutak/auv/logs/slurm/%x-%j.out
#SBATCH --error=/ari/users/btutak/auv/logs/slurm/%x-%j.err

# Distributed roll-task training across all 4 A100s on one node using
# torchrun + rsl_rl's built-in NCCL process-group setup. Each rank gets its
# own GPU (LOCAL_RANK), seed offset, and EGL device id.

set -euo pipefail

AUV_HOME="${AUV_HOME:-$HOME/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

mkdir -p "$AUV_HOME/.ssh" "$AUV_HOME/.config" "$AUV_HOME/.cache" "$AUV_HOME/tmp" "$AUV_HOME/bin"
export HOME="$AUV_HOME"
export USER="${USER:-btutak}"
export LOGNAME="${LOGNAME:-$USER}"
export TMPDIR="$AUV_HOME/tmp"
export XDG_CONFIG_HOME="$AUV_HOME/.config"
export XDG_CACHE_HOME="$AUV_HOME/.cache"
export GIT_CONFIG_GLOBAL="$AUV_HOME/.gitconfig"
export GIT_SSH_COMMAND="ssh -F $AUV_HOME/.ssh/config"
export HISTFILE="$AUV_HOME/.bash_history"
export PATH="$AUV_HOME/bin:/ari/progs/uhem-bin:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$AUV_HOME/.cache/uv}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$AUV_HOME/.cache/matplotlib}"
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER CONDA_SHLVL
unset VIRTUAL_ENV PYTHONHOME PYTHONPATH

cd "$AUV_PROJECT_DIR"

mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$AUV_HOME/logs/slurm" logs/remote_runs

if [ -n "${RUN_LOG_PATH:-}" ]; then
  mkdir -p "$(dirname "$RUN_LOG_PATH")"
  exec > >(tee -a "$RUN_LOG_PATH") 2>&1
fi

STAGE="${STAGE:-c0_90_discovery}"
ITERS="${ITERS:-700}"
NUM_ENVS_PER_GPU="${NUM_ENVS_PER_GPU:-1024}"
NUM_STEPS_PER_ENV="${NUM_STEPS_PER_ENV:-256}"
SAVE_INTERVAL="${SAVE_INTERVAL:-50}"
NUM_GPUS="${NUM_GPUS:-4}"
RUN_NAME="${RUN_NAME:-${STAGE}_slurm_${SLURM_JOB_ID:-manual}}"
RESUME_MODE="${RESUME_MODE:-weights-only}"

# All ranks must share the SAME log directory; pin it here before launching torchrun.
EXP_NAME="${EXP_NAME:-taluy_roll_v1}"
LOG_STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
export AUVRL_LOG_DIR="$AUV_PROJECT_DIR/logs/rsl_rl/${EXP_NAME}/${LOG_STAMP}_${RUN_NAME}"
mkdir -p "$AUVRL_LOG_DIR"

# Port for torchrun rendezvous (single-node so MASTER_ADDR is localhost).
MASTER_PORT="${MASTER_PORT:-$((25000 + (SLURM_JOB_ID % 10000)))}"

cmd=(
  uv run python -u -m torch.distributed.run
  --standalone
  --nproc-per-node="$NUM_GPUS"
  --master-port="$MASTER_PORT"
  -m auvrl.scripts.train.taluy_roll
  --device cuda
  --curriculum-stage "$STAGE"
  --iterations "$ITERS"
  --num-envs "$NUM_ENVS_PER_GPU"
  --num-steps-per-env "$NUM_STEPS_PER_ENV"
  --save-interval "$SAVE_INTERVAL"
  --run-name "$RUN_NAME"
)

if [ -n "${LEARNING_RATE:-}" ]; then
  cmd+=(--learning-rate "$LEARNING_RATE")
fi

if [ -n "${ENTROPY_COEF:-}" ]; then
  cmd+=(--entropy-coef "$ENTROPY_COEF")
fi

if [ -n "${DESIRED_KL:-}" ]; then
  cmd+=(--desired-kl "$DESIRED_KL")
fi

if [ -n "${RESUME_CHECKPOINT:-}" ]; then
  cmd+=(--resume-checkpoint "$RESUME_CHECKPOINT" --resume-mode "$RESUME_MODE")
fi

printf 'host=%s\n' "$(hostname)"
printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'stage=%s\n' "$STAGE"
printf 'run_name=%s\n' "$RUN_NAME"
printf 'num_gpus=%s envs_per_gpu=%s total_envs=%s\n' "$NUM_GPUS" "$NUM_ENVS_PER_GPU" "$((NUM_GPUS * NUM_ENVS_PER_GPU))"
printf 'resume_checkpoint=%s\n' "${RESUME_CHECKPOINT:-}"
printf 'resume_mode=%s\n' "$RESUME_MODE"
printf 'log_dir=%s\n' "$AUVRL_LOG_DIR"
printf 'master_port=%s\n' "$MASTER_PORT"
printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
printf 'gpus_visible=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
