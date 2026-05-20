from __future__ import annotations

import argparse

import numpy as np

from auvrl.scripts.demo import model_replay
from auvrl.scripts.demo._model_replay_io import FORMAT_VERSION, load_recording


def test_replay_timeline_selects_frames_by_rtf() -> None:
    frame_indices = np.asarray([0, 1, 2, 3], dtype=np.int64)
    time_s = np.asarray([0.0, 0.008, 0.016, 0.024], dtype=np.float64)
    timeline = model_replay.ReplayTimeline(
        frame_indices,
        time_s,
        rtf=1.0,
        frame_stride=1,
        start_paused=True,
        loop=False,
    )

    assert timeline.current_frame == 0
    timeline.set_local_cursor(2)
    assert timeline.current_frame == 2
    timeline.toggle_pause()
    timeline.toggle_pause()
    assert timeline.paused


def test_zero_policy_record_smoke(tmp_path) -> None:
    from auvrl.scripts.demo import model_record

    args = argparse.Namespace(
        policy="zero",
        device="cpu",
        num_envs=1,
        record_env_idx=0,
        episodes=1,
        max_steps=3,
        post_done_steps=0,
        output_dir=tmp_path,
        overwrite=True,
        curriculum_stage="c0_90_discovery",
        play_mode="deployment",
        deployment_completion_action="zero",
        episode_length_s=None,
        roll_direction=1,
        print_period=0,
    )

    device = model_record._resolve_device(args.device)
    monitor_params = model_record._make_monitor_params(args)
    env = model_record.RslRlVecEnvWrapper(
        env=model_record._make_env(args, device),
        clip_actions=1.0,
    )
    try:
        policy = model_record._make_policy(args, env, None, {}, device)
        recorder = model_record.EpisodeRecorder(
            env.unwrapped,
            env_idx=0,
            policy_name="zero",
            checkpoint_path=None,
            curriculum_stage=args.curriculum_stage,
            play_mode=args.play_mode,
            deployment_completion_action=args.deployment_completion_action,
            monitor_params=monitor_params,
        )
        env.reset()
        for episode_idx in range(1):
            for _ in range(3):
                obs = env.get_observations()
                actions = recorder.filter_actions(policy(obs))
                (
                    obs_after,
                    _reward,
                    terminated,
                    truncated,
                    _extras,
                ) = model_record.step_vec_env(env, actions)
                recorder.capture(
                    episode_idx=episode_idx,
                    action=actions,
                    obs=obs_after,
                    terminated=terminated,
                    truncated=truncated,
                )
        arrays = recorder.arrays()
        model_record.write_recording(
            tmp_path,
            manifest={
                "format": FORMAT_VERSION,
                "control_dt_s": float(env.unwrapped.step_dt),
                "reward_term_names": list(env.unwrapped.reward_manager.active_terms),
            },
            arrays=arrays,
            episode_summaries=[{"episode_idx": 0, "frames": 3}],
            overwrite=True,
        )
    finally:
        env.close()

    loaded = load_recording(tmp_path)
    assert loaded.num_frames == 3
    assert loaded.arrays["actor_obs"].shape == (3, 15)
    assert loaded.arrays["critic_obs"].shape == (3, 20)
    assert loaded.arrays["action"].shape == (3, 6)
