import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import dill
import math
import wandb.sdk.data_types.video as wv
import os
import time
import traceback

from diffusion_policy.env.pusht.pusht_keypoints_env import PushTKeypointsEnv
# from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv as AsyncVectorEnv
# from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder

from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner


def _ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _rlog(msg):
    print(f"[{_ts()}] [PUSHT_RUNNER] {msg}", flush=True)


class _Timer:
    def __init__(self, name):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        _rlog(f"[TIMER] start {self.name}")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        _rlog(f"[TIMER] end   {self.name} dt={dt:.4f}s")
        return False  # never swallow exceptions


def _safe_shape(x):
    try:
        return tuple(x.shape)
    except Exception:
        return None


class PushTKeypointsRunner(BaseLowdimRunner):
    def __init__(self,
            output_dir,
            keypoint_visible_rate=1.0,
            n_train=10,
            n_train_vis=3,
            train_start_seed=0,
            n_test=22,
            n_test_vis=6,
            legacy_test=False,
            test_start_seed=10000,
            max_steps=200,
            n_obs_steps=8,
            n_action_steps=8,
            n_latency_steps=0,
            fps=10,
            crf=22,
            agent_keypoints=False,
            past_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None
        ):
        super().__init__(output_dir)

        _rlog(f"__init__ begin output_dir={output_dir}")
        _rlog(f"cfg n_train={n_train} n_test={n_test} n_envs={n_envs} "
              f"n_train_vis={n_train_vis} n_test_vis={n_test_vis} "
              f"train_start_seed={train_start_seed} test_start_seed={test_start_seed} "
              f"max_steps={max_steps} n_obs_steps={n_obs_steps} n_action_steps={n_action_steps} n_latency_steps={n_latency_steps} "
              f"fps={fps} crf={crf} past_action={past_action} agent_keypoints={agent_keypoints} legacy_test={legacy_test}")

        if n_envs is None:
            n_envs = n_train + n_test
        _rlog(f"resolved n_envs={n_envs}")

        env_n_obs_steps = n_obs_steps + n_latency_steps
        env_n_action_steps = n_action_steps

        kp_kwargs = PushTKeypointsEnv.genenerate_keypoint_manager_params()
        _rlog(f"kp_kwargs keys={list(kp_kwargs.keys()) if isinstance(kp_kwargs, dict) else type(kp_kwargs)}")

        def env_fn():
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    PushTKeypointsEnv(
                        legacy=legacy_test,
                        keypoint_visible_rate=keypoint_visible_rate,
                        agent_keypoints=agent_keypoints,
                        **kp_kwargs
                    ),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                ),
                n_obs_steps=env_n_obs_steps,
                n_action_steps=env_n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        for i in range(n_train):
            seed = train_start_seed + i
            enable_render = i < n_train_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('train/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed, enable_render=enable_render):
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                assert isinstance(env, MultiStepWrapper)
                env.seed(seed)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        _rlog(f"prepared init fns: n_train={n_train} n_test={n_test} total_inits={len(env_init_fn_dills)}")
        _rlog("creating AsyncVectorEnv ... (if hang happens here, it’s multiprocessing init)")

        with _Timer("AsyncVectorEnv(env_fns)"):
            env = AsyncVectorEnv(env_fns)

        _rlog("AsyncVectorEnv created")

        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.agent_keypoints = agent_keypoints
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.n_latency_steps = n_latency_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.tqdm_interval_sec = tqdm_interval_sec

        _rlog("__init__ done")

    def run(self, policy: BaseLowdimPolicy):
        _rlog("run() begin")
        device = policy.device
        dtype = policy.dtype
        _rlog(f"policy device={device} dtype={dtype} policy_cls={policy.__class__}")

        env = self.env

        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        _rlog(f"rollout plan: n_envs={n_envs} n_inits={n_inits} n_chunks={n_chunks} max_steps={self.max_steps} "
              f"n_obs_steps={self.n_obs_steps} n_action_steps={self.n_action_steps} n_latency_steps={self.n_latency_steps} past_action={self.past_action}")

        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0, this_n_active_envs)

            _rlog(f"chunk {chunk_idx+1}/{n_chunks} start: global_slice=({start}:{end}) active_envs={this_n_active_envs}")

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]] * n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            try:
                with _Timer(f"chunk {chunk_idx+1}: env.call_each(run_dill_function)"):
                    env.call_each('run_dill_function',
                                  args_list=[(x,) for x in this_init_fns])
                _rlog(f"chunk {chunk_idx+1}: call_each done")
            except Exception as e:
                _rlog(f"EXCEPTION in env.call_each: {repr(e)}")
                _rlog(traceback.format_exc())
                raise

            # start rollout
            try:
                with _Timer(f"chunk {chunk_idx+1}: env.reset()"):
                    obs = env.reset()
                _rlog(f"chunk {chunk_idx+1}: reset returned obs.shape={_safe_shape(obs)} obs.dtype={getattr(obs,'dtype',None)} obs.type={type(obs)}")
            except Exception as e:
                _rlog(f"EXCEPTION in env.reset: {repr(e)}")
                _rlog(traceback.format_exc())
                raise

            past_action = None
            try:
                policy.reset()
                _rlog(f"chunk {chunk_idx+1}: policy.reset() done")
            except Exception as e:
                _rlog(f"EXCEPTION in policy.reset: {repr(e)}")
                _rlog(traceback.format_exc())
                raise

            pbar = tqdm.tqdm(
                total=self.max_steps,
                desc=f"Eval PushtKeypointsRunner {chunk_idx+1}/{n_chunks}",
                leave=False,
                mininterval=self.tqdm_interval_sec
            )
            _rlog(f"chunk {chunk_idx+1}: tqdm created total={self.max_steps}")

            done = False
            iter_idx = 0
            while not done:
                iter_idx += 1
                if iter_idx <= 3:
                    _rlog(f"chunk {chunk_idx+1}: while iter={iter_idx} obs.shape={_safe_shape(obs)}")

                Do = obs.shape[-1] // 2

                np_obs_dict = {
                    'obs': obs[...,:self.n_obs_steps,:Do].astype(np.float32),
                    'obs_mask': obs[...,:self.n_obs_steps,Do:] > 0.5
                }
                if self.past_action and (past_action is not None):
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                try:
                    with _Timer(f"chunk {chunk_idx+1}: obs -> torch (iter={iter_idx})"):
                        obs_dict = dict_apply(
                            np_obs_dict,
                            lambda x: torch.from_numpy(x).to(device=device)
                        )
                except Exception as e:
                    _rlog(f"EXCEPTION in obs transfer: {repr(e)}")
                    _rlog(traceback.format_exc())
                    raise

                try:
                    with torch.no_grad():
                        with _Timer(f"chunk {chunk_idx+1}: policy.predict_action (iter={iter_idx})"):
                            action_dict = policy.predict_action(obs_dict)
                except Exception as e:
                    _rlog(f"EXCEPTION in policy.predict_action: {repr(e)}")
                    _rlog(traceback.format_exc())
                    raise

                try:
                    with _Timer(f"chunk {chunk_idx+1}: action -> numpy (iter={iter_idx})"):
                        np_action_dict = dict_apply(
                            action_dict,
                            lambda x: x.detach().to('cpu').numpy()
                        )
                except Exception as e:
                    _rlog(f"EXCEPTION in action transfer: {repr(e)}")
                    _rlog(traceback.format_exc())
                    raise

                action = np_action_dict['action'][:, self.n_latency_steps:]
                if iter_idx <= 3:
                    _rlog(f"chunk {chunk_idx+1}: action.shape={_safe_shape(action)} latency={self.n_latency_steps}")

                try:
                    with _Timer(f"chunk {chunk_idx+1}: env.step (iter={iter_idx})"):
                        obs, reward, done, info = env.step(action)
                    if iter_idx <= 3:
                        _rlog(f"chunk {chunk_idx+1}: step returned obs.shape={_safe_shape(obs)} "
                              f"reward.shape={_safe_shape(reward)} done.shape={_safe_shape(done)} done_sample={done}")
                except Exception as e:
                    _rlog(f"EXCEPTION in env.step: {repr(e)}")
                    _rlog(traceback.format_exc())
                    raise

                done = np.all(done)
                past_action = action

                pbar.update(action.shape[1])

            pbar.close()
            _rlog(f"chunk {chunk_idx+1}: rollout loop done")

            try:
                with _Timer(f"chunk {chunk_idx+1}: env.render()"):
                    rendered = env.render()
                _rlog(f"chunk {chunk_idx+1}: env.render returned type={type(rendered)} len={len(rendered) if hasattr(rendered,'__len__') else None}")
            except Exception as e:
                _rlog(f"EXCEPTION in env.render: {repr(e)}")
                _rlog(traceback.format_exc())
                raise

            try:
                with _Timer(f"chunk {chunk_idx+1}: env.call(get_attr, reward)"):
                    rewards = env.call('get_attr', 'reward')
                _rlog(f"chunk {chunk_idx+1}: env.call(get_attr,reward) returned type={type(rewards)} len={len(rewards) if hasattr(rewards,'__len__') else None}")
            except Exception as e:
                _rlog(f"EXCEPTION in env.call(get_attr,reward): {repr(e)}")
                _rlog(traceback.format_exc())
                raise

            all_video_paths[this_global_slice] = rendered[this_local_slice]
            all_rewards[this_global_slice] = rewards[this_local_slice]

        max_rewards = collections.defaultdict(list)
        log_data = dict()

        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward

            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value

        _rlog(f"run() end returning keys={list(log_data.keys())[:20]} (total={len(log_data)})")
        return log_data
