# AUVRL

AUVRL is a small research codebase for training underwater robot control policies in MJLab/MuJoCo. It currently focuses on Taluy, with environments for 6-DoF velocity tracking and roll-maneuver learning. The project keeps the robot model, actuator/thruster code, task definitions, PPO configs, training scripts, and playback tools in one place.

The code is meant for quick local iteration: run a smoke test, train a short PPO job, inspect the policy in a viewer, then scale the run when the behavior looks sane.

## Setup

Use Python 3.10+ and `uv`.

```bash
git clone git@github.com:eminmeydanoglu/auv-rl.git
cd auv-rl
uv sync
```

If you are running headless, the training scripts default MuJoCo to EGL. CUDA is used automatically when available; otherwise the scripts fall back to CPU with fewer parallel environments.

Run a quick environment check:

```bash
uv run python -m auvrl.scripts.smoke.taluy_roll_env
uv run python -m auvrl.scripts.smoke.taluy_velocity_env
```

## Train

Roll task, short smoke run:

```bash
uv run python -m auvrl.scripts.train.taluy_roll \
  --curriculum-stage c0_90_discovery \
  --iterations 50 \
  --run-name c0-smoke
```

Velocity task, short smoke run:

```bash
uv run python -m auvrl.scripts.train.taluy_velocity \
  --iterations 50 \
  --run-name velocity-smoke
```

Training outputs go under:

```text
logs/rsl_rl/<experiment>/<timestamp>_<run_name>/
```

That folder contains checkpoints, TensorBoard events, and the exact env/agent YAML used for the run. These files are local artifacts and are intentionally ignored by git.

Useful training flags:

```bash
--device auto|cuda|cpu
--num-envs 256
--iterations 1000
--num-steps-per-env 48
--save-interval 100
--resume-checkpoint logs/rsl_rl/.../model_*.pt
```

For the roll task, curriculum stages are trained explicitly:

```text
c0_90_discovery
c1_180_inversion
c2_360_full_turn
c2a_360_reach
c3_720_no_hard_stop
c4_720_settle
```

## Play And Inspect

Open the roll inspector manually, without a checkpoint:

```bash
uv run python -m auvrl.scripts.demo.taluy_roll_play \
  --curriculum-stage c0_90_discovery \
  --policy manual \
  --viewer viser
```

Play a trained roll checkpoint:

```bash
uv run python -m auvrl.scripts.demo.taluy_roll_play \
  --policy checkpoint \
  --checkpoint-file logs/rsl_rl/taluy_roll_v1/<run>/model_*.pt \
  --viewer viser
```

Play a trained velocity checkpoint:

```bash
uv run python -m auvrl.scripts.demo.taluy_velocity_play \
  --checkpoint-file logs/rsl_rl/taluy_velocity_6d/<run>/model_*.pt \
  --viewer viser
```

The Viser viewer prints a local URL when it starts. Use it for live inspection, command inputs, and policy debugging.

## Repository Layout

```text
src/auvrl/actuator/        Thruster and body-wrench action code
src/auvrl/asset_zoo/       Taluy model files and thruster configs
src/auvrl/config/          Shared AUV and thruster configuration
src/auvrl/envs/            Base MJLab environment pieces
src/auvrl/tasks/roll/      Roll task, rewards, observations, curriculum
src/auvrl/tasks/velocity/  6-DoF velocity tracking task
src/auvrl/scripts/train/   PPO training entrypoints
src/auvrl/scripts/demo/    Playback and visual inspection tools
src/auvrl/scripts/smoke/   Fast sanity checks
tests/                     Unit and integration tests
```

## Tests

Run the Python tests with:

```bash
uv run --with pytest pytest
```

For quick task-level checks while iterating:

```bash
uv run --with pytest pytest tests/tasks/roll
```

## Notes

This is research code, so most outputs are disposable by design. Keep runs, screenshots, reports, local scripts, and checkpoints out of git unless they are deliberately promoted into a documented artifact.
