#!/usr/bin/env bash
#SBATCH -J roll-select-A-best
#SBATCH -A kds3by
#SBATCH -p a100q
#SBATCH --gres=gpu:a100:1
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --time=02:00:00
#SBATCH --output=/ari/users/btutak/auv/logs/slurm/%x-%j.out
#SBATCH --error=/ari/users/btutak/auv/logs/slurm/%x-%j.err

set -euo pipefail

AUV_HOME="${AUV_HOME:-/ari/users/btutak/auv}"
AUV_PROJECT_DIR="${AUV_PROJECT_DIR:-$AUV_HOME/auv-rl}"

mkdir -p "$AUV_HOME/.ssh" "$AUV_HOME/.config" "$AUV_HOME/.cache" "$AUV_HOME/tmp" "$AUV_HOME/bin"
chmod 700 "$AUV_HOME" "$AUV_HOME/.ssh"

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
mkdir -p "$UV_CACHE_DIR" "$MPLCONFIGDIR" "$AUV_HOME/logs/slurm" logs/checkpoint_eval logs/remote_runs

A_JOB_ID="${A_JOB_ID:-}"
A_RUN_DIR="${A_RUN_DIR:-}"
A_STAGE="${A_STAGE:-c2b_360_hold_0p05}"
B_STAGE="${B_STAGE:-c2c_360_hold_0p10}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-1024}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda}"
CANDIDATE_ITERS="${CANDIDATE_ITERS:-200 300 450 600 699}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [ -z "$A_RUN_DIR" ] && [ -n "$A_JOB_ID" ]; then
  A_LOG="$AUV_HOME/logs/slurm/roll-A-c2b-${A_JOB_ID}.out"
  if [ -f "$A_LOG" ]; then
    A_RUN_DIR="$(grep -E '^log_dir=' "$A_LOG" | tail -n 1 | cut -d= -f2-)"
  fi
fi

if [ -z "$A_RUN_DIR" ]; then
  A_RUN_DIR="$(
    find logs/rsl_rl/taluy_roll_v1 -maxdepth 1 -type d -name '*c2b_360_hold_0p05_from_c2a699*' \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -n 1 \
      | cut -d' ' -f2-
  )"
fi

if [ -z "$A_RUN_DIR" ] || [ ! -d "$A_RUN_DIR" ]; then
  echo "Could not resolve A run directory. A_JOB_ID=$A_JOB_ID A_RUN_DIR=$A_RUN_DIR" >&2
  exit 2
fi

SUMMARY_JSON="logs/checkpoint_eval/b_from_a_selector_${STAMP}_summary.json"
JSONL="logs/checkpoint_eval/b_from_a_selector_${STAMP}.jsonl"
BEST_TXT="logs/checkpoint_eval/b_from_a_selector_${STAMP}_best.txt"

printf 'selector_job_id=%s\n' "${SLURM_JOB_ID:-manual}"
printf 'a_job_id=%s\n' "$A_JOB_ID"
printf 'a_run_dir=%s\n' "$A_RUN_DIR"
printf 'a_stage=%s\n' "$A_STAGE"
printf 'b_stage=%s\n' "$B_STAGE"
printf 'candidate_iters=%s\n' "$CANDIDATE_ITERS"
printf 'started_at=%s\n' "$(date --iso-8601=seconds)"

A_RUN_DIR="$A_RUN_DIR" \
A_STAGE="$A_STAGE" \
CANDIDATE_ITERS="$CANDIDATE_ITERS" \
EVAL_NUM_ENVS="$EVAL_NUM_ENVS" \
EVAL_DEVICE="$EVAL_DEVICE" \
SUMMARY_JSON="$SUMMARY_JSON" \
JSONL="$JSONL" \
BEST_TXT="$BEST_TXT" \
uv run python - <<'PY'
from __future__ import annotations

from dataclasses import asdict
import json
import math
import os
from pathlib import Path
from typing import Any

import torch
import yaml

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from auvrl import (
    get_roll_curriculum_stage,
    make_taluy_roll_env_cfg,
    taluy_roll_ppo_runner_cfg,
)
from auvrl.tasks.roll import mdp


def load_agent_cfg(checkpoint_path: Path) -> dict[str, Any]:
    params_path = checkpoint_path.parent / "params" / "agent.yaml"
    if not params_path.exists():
        return asdict(taluy_roll_ppo_runner_cfg())
    with params_path.open("r", encoding="utf-8") as file:
        data = yaml.full_load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {params_path}, got {type(data).__name__}")
    return data


def mean_of(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any().item()):
        return 0.0
    return float(values[mask].float().mean().detach().cpu().item())


def cfg_step_dt(cfg: Any) -> float:
    return float(cfg.sim.mujoco.timestep) * float(cfg.decimation)


def evaluate(checkpoint_path: Path, stage_name: str, num_envs: int, device: str) -> dict[str, Any]:
    stage = get_roll_curriculum_stage(stage_name)
    cfg = make_taluy_roll_env_cfg(num_envs=num_envs, curriculum_stage=stage.name)
    task_success_params = dict(cfg.terminations["task_success"].params)
    fail_pitch_limit = float(cfg.terminations["excess_pitch"].params["limit_rad"])
    fail_depth_limit = float(cfg.terminations["excess_depth_error"].params["limit_m"])
    fail_xy_limit = float(cfg.terminations["excess_xy_drift"].params["limit_m"])
    max_steps = int(math.ceil(float(cfg.episode_length_s) / cfg_step_dt(cfg)))
    cfg.terminations = {}

    base_env: ManagerBasedRlEnv | None = None
    env: RslRlVecEnvWrapper | None = None
    try:
        base_env = ManagerBasedRlEnv(cfg=cfg, device=device)
        agent_cfg = load_agent_cfg(checkpoint_path)
        env = RslRlVecEnvWrapper(base_env, clip_actions=float(agent_cfg.get("clip_actions", 1.0)))
        runner = MjlabOnPolicyRunner(env, agent_cfg, device=device)
        runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)
        obs, _ = base_env.reset()

        done = torch.zeros(base_env.num_envs, dtype=torch.bool, device=base_env.device)
        event_step = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
        event_kind = torch.full((base_env.num_envs,), -1, dtype=torch.long, device=base_env.device)
        reward_sum = torch.zeros(base_env.num_envs, dtype=torch.float, device=base_env.device)

        final_progress = torch.zeros(base_env.num_envs, dtype=torch.float, device=base_env.device)
        final_phi = torch.zeros_like(final_progress)
        final_settle_s = torch.zeros_like(final_progress)
        final_depth = torch.zeros_like(final_progress)
        final_xy = torch.zeros_like(final_progress)
        final_pitch = torch.zeros_like(final_progress)
        final_yaw = torch.zeros_like(final_progress)
        final_ang_speed = torch.zeros_like(final_progress)

        for step in range(max_steps):
            with torch.no_grad():
                action = policy(obs)
                obs, reward, _, _ = env.step(action)
                reward_sum += reward

                success = mdp.roll_task_success(base_env, **task_success_params)
                fail_pitch = mdp.excess_pitch(base_env, limit_rad=fail_pitch_limit)
                fail_depth = mdp.excess_depth_error(base_env, limit_m=fail_depth_limit)
                fail_xy = mdp.excess_xy_drift(base_env, limit_m=fail_xy_limit)
                timeout = torch.full_like(success, step == max_steps - 1)
                event = (~done) & (success | fail_pitch | fail_depth | fail_xy | timeout)

                if bool(event.any().item()):
                    final_progress[event] = mdp.roll_progress_ratio(
                        base_env,
                        target_roll_rad=task_success_params["target_roll_rad"],
                        roll_direction=task_success_params["roll_direction"],
                    )[event]
                    final_phi[event] = mdp.phi_total_rad(base_env)[event]
                    final_settle_s[event] = mdp.settle_counter_s(base_env)[event]
                    final_depth[event] = mdp.depth_abs_error_m(base_env)[event]
                    final_xy[event] = mdp.xy_drift_m(base_env)[event]
                    final_pitch[event] = mdp.pitch_abs_rad(base_env)[event]
                    final_yaw[event] = mdp.yaw_abs_error_rad(base_env)[event]
                    final_ang_speed[event] = mdp.root_ang_speed_rad_s(base_env)[event]
                    event_step[event] = step
                    event_kind[event & success] = 0
                    event_kind[event & ~success & fail_pitch] = 1
                    event_kind[event & ~success & ~fail_pitch & fail_depth] = 2
                    event_kind[event & ~success & ~fail_pitch & ~fail_depth & fail_xy] = 3
                    event_kind[event & ~success & ~fail_pitch & ~fail_depth & ~fail_xy & timeout] = 4
                    done |= event
                    if bool(done.all().item()):
                        break

        success_mask = event_kind == 0
        fail_pitch_mask = event_kind == 1
        fail_depth_mask = event_kind == 2
        fail_xy_mask = event_kind == 3
        timeout_mask = event_kind == 4
        success_rate = mean_of(success_mask.float(), torch.ones_like(success_mask, dtype=torch.bool))
        progress_mean = mean_of(final_progress, torch.ones_like(done, dtype=torch.bool))
        score = (
            success_rate * 10000.0
            + min(progress_mean, 1.25) * 150.0
            + mean_of(final_settle_s, torch.ones_like(done, dtype=torch.bool)) * 200.0
            - mean_of(final_pitch, torch.ones_like(done, dtype=torch.bool)) * 70.0
            - mean_of(final_yaw, torch.ones_like(done, dtype=torch.bool)) * 30.0
            - mean_of(final_depth, torch.ones_like(done, dtype=torch.bool)) * 30.0
            - mean_of(final_xy, torch.ones_like(done, dtype=torch.bool)) * 10.0
            - mean_of(final_ang_speed, torch.ones_like(done, dtype=torch.bool)) * 5.0
        )

        return {
            "checkpoint": str(checkpoint_path),
            "stage": stage.name,
            "score": score,
            "success_rate": success_rate,
            "counts": {
                "success": int(success_mask.sum().detach().cpu().item()),
                "fail_pitch": int(fail_pitch_mask.sum().detach().cpu().item()),
                "fail_depth": int(fail_depth_mask.sum().detach().cpu().item()),
                "fail_xy": int(fail_xy_mask.sum().detach().cpu().item()),
                "timeout": int(timeout_mask.sum().detach().cpu().item()),
            },
            "event_step_mean": mean_of(event_step.float(), event_step >= 0),
            "success_time_s_mean": mean_of(event_step.float(), success_mask) * float(base_env.step_dt),
            "progress_mean": progress_mean,
            "phi_mean": mean_of(final_phi, torch.ones_like(done, dtype=torch.bool)),
            "settle_s_mean": mean_of(final_settle_s, torch.ones_like(done, dtype=torch.bool)),
            "depth_m_mean": mean_of(final_depth, torch.ones_like(done, dtype=torch.bool)),
            "xy_m_mean": mean_of(final_xy, torch.ones_like(done, dtype=torch.bool)),
            "pitch_rad_mean": mean_of(final_pitch, torch.ones_like(done, dtype=torch.bool)),
            "yaw_rad_mean": mean_of(final_yaw, torch.ones_like(done, dtype=torch.bool)),
            "ang_speed_rad_s_mean": mean_of(final_ang_speed, torch.ones_like(done, dtype=torch.bool)),
            "reward_sum_mean": mean_of(reward_sum, torch.ones_like(done, dtype=torch.bool)),
            "num_envs": int(base_env.num_envs),
            "max_steps": max_steps,
            "step_dt": float(base_env.step_dt),
        }
    finally:
        if env is not None:
            env.close()
        elif base_env is not None:
            base_env.close()


a_run_dir = Path(os.environ["A_RUN_DIR"])
stage = os.environ["A_STAGE"]
num_envs = int(os.environ["EVAL_NUM_ENVS"])
device = os.environ["EVAL_DEVICE"]
candidate_iters = [item for item in os.environ["CANDIDATE_ITERS"].split() if item]
candidates: list[Path] = []
for item in candidate_iters:
    path = a_run_dir / f"model_{item}.pt"
    if path.exists():
        candidates.append(path)
if not candidates:
    candidates = sorted(
        (path for path in a_run_dir.glob("model_*.pt") if path.name != "model_0.pt"),
        key=lambda path: int(path.stem.split("_", 1)[1]),
    )
if not candidates:
    raise SystemExit(f"No candidate checkpoints found under {a_run_dir}")

jsonl_path = Path(os.environ["JSONL"])
summary_path = Path(os.environ["SUMMARY_JSON"])
best_path = Path(os.environ["BEST_TXT"])
rows = []
with jsonl_path.open("w", encoding="utf-8") as jsonl:
    for checkpoint in candidates:
        row = evaluate(checkpoint, stage, num_envs, device)
        rows.append(row)
        jsonl.write(json.dumps(row, sort_keys=True) + "\n")
        jsonl.flush()

best = max(rows, key=lambda row: row["score"])
summary = {"best_checkpoint": best["checkpoint"], "best": best, "candidates": rows}
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
best_path.write_text(best["checkpoint"] + "\n", encoding="utf-8")
print(best["checkpoint"])
PY
BEST_CHECKPOINT="$(cat "$BEST_TXT")"

printf 'best_checkpoint=%s\n' "$BEST_CHECKPOINT"
printf 'summary_json=%s\n' "$SUMMARY_JSON"
printf 'jsonl=%s\n' "$JSONL"

B_RUN_NAME="${B_RUN_NAME:-${B_STAGE}_from_c2b_best_env1024_steps256_${STAMP}}"
B_SUBMISSIONS="logs/remote_runs/roll_b_from_a_best_submission_${STAMP}.tsv"

submit_output="$(
  sbatch \
    --job-name "roll-B-from-A-best" \
    --output "$AUV_HOME/logs/slurm/%x-%j.out" \
    --error "$AUV_HOME/logs/slurm/%x-%j.err" \
    --export "ALL,AUV_HOME=$AUV_HOME,AUV_PROJECT_DIR=$AUV_PROJECT_DIR,RESUME_CHECKPOINT=$BEST_CHECKPOINT,RESUME_MODE=weights-only,STAGE=$B_STAGE,RUN_NAME=$B_RUN_NAME,NUM_ENVS=1024,NUM_STEPS_PER_ENV=256,ITERS=700,SAVE_INTERVAL=50,LEARNING_RATE=3e-4,ENTROPY_COEF=0.003,DESIRED_KL=0.006" \
    jobs/slurm/taluy_roll.sbatch
)"
B_JOB_ID="$(grep -Eo '[0-9]+' <<<"$submit_output" | tail -n 1)"

{
  printf 'job_id\tjob_name\tstage\trun_name\tresume_checkpoint\tselector_job_id\tsummary_json\n'
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$B_JOB_ID" \
    "roll-B-from-A-best" \
    "$B_STAGE" \
    "$B_RUN_NAME" \
    "$BEST_CHECKPOINT" \
    "${SLURM_JOB_ID:-manual}" \
    "$SUMMARY_JSON"
} > "$B_SUBMISSIONS"

printf 'submitted_b_job_id=%s\n' "$B_JOB_ID"
printf 'submission_tsv=%s\n' "$B_SUBMISSIONS"
