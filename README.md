# AUVRL

AUVRL is a codebase for training underwater robot control policies in MJLab/MuJoCo. It includes the robot [Taluy](https://auv.itu.edu.tr/vehicle-taluy.html), with training environments for a 6-DoF velocity controller and a roll-maneuver specialist controller (for this [task](https://robonation.gitbook.io/robosub-resources/section-3-autonomy-challenge/3.2-task-descriptions#:~:text=Figure%20%3A%20Heading%20Out-,3.2.2%20Task%201%20%2D%20Begin%20Assessment%20(Gate),-Head%20out%20to) in RoboSub). Robot model, actuator/thruster code, task definitions, PPO configs, training scripts, curriculums/ data randomization, playback and debug tools in one place.



## Setup

Use Python 3.10+ and `uv`.

```bash
git clone git@github.com:eminmeydanoglu/auv-rl.git
cd auv-rl
uv sync
```

Run an environment check:

```bash
uv run python -m auvrl.scripts.smoke.taluy_roll_env
uv run python -m auvrl.scripts.smoke.taluy_velocity_env
```



## Play And Inspect

It is easy to visually test how good the agent is performing.

![Taluy roll policy demo](docs/media/balerina.webp)



Play a trained checkpoint:

```bash
uv run python -m auvrl.scripts.demo.taluy_roll_play \
  --policy checkpoint \
  --checkpoint-file logs/rsl_rl/taluy_roll_v1/<run>/model_*.pt \
  --viewer viser
```

Or just pop up the simulation and send manual control inputs - to debug observations, rewards and to see how the robot behaves.

```bash
uv run python -m auvrl.scripts.demo.taluy_roll_play \
  --curriculum-stage c0_90_discovery \
  --policy manual \
  --viewer viser
```



## Train

Roll task short smoke run:

```bash
uv run python -m auvrl.scripts.train.taluy_roll \
  --curriculum-stage c0_90_discovery \
  --iterations 50 \
  --run-name c0-smoke
```

Velocity task short smoke run:

```bash
uv run python -m auvrl.scripts.train.taluy_velocity \
  --iterations 50 \
  --run-name velocity-smoke
```

Other useful training flags:

```bash
--device auto|cuda|cpu
--num-envs 256
--iterations 1000
--num-steps-per-env 48
--save-interval 100
--resume-checkpoint logs/rsl_rl/.../model_*.pt
```
Curriculum stages/ domain randomization exists! For example, for the roll specialist agent; there are

```text
c0_90_discovery
c1_180_inversion
c2_360_full_turn
c2a_360_reach
c3_720_no_hard_stop
c4_720_settle
```
