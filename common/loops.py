#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : loops.py
# @Author: harry
# @Date  : 2/25/21 9:01 PM
# @Desc  : Common training and evaluating loops

import tqdm
import cv2
import numpy as np
import matplotlib.pyplot as plt
import vizdoom as vzd

from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, BasePolicy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from common.game_wrapper import DoomEnv
from common.utils import linear_schedule, collect_kv, get_img_from_fig
from common.evaluate_recurrent_policy import RecurrentEvalCallback
from common.i_reward_shaper import IRewardShaper
from common.vec_curiosity_wrapper_rdn import RdnWrapper
from common.vec_curiosity_wrapper_icm import IcmWrapper
from common.policies import AugmentedCnnLstmPolicy
from common.larger_policies import AugLargerLnCnnLstmPolicy
from common.augmented_ppo2 import AugmentedPPO2
from typing import Dict, Tuple, Any, Type, List, Union, Optional


RECURRENT_POLICIES = (CnnLstmPolicy, CnnLnLstmPolicy, AugmentedCnnLstmPolicy, AugLargerLnCnnLstmPolicy)
AUGMENTED_POLICIES = (AugmentedCnnLstmPolicy, AugLargerLnCnnLstmPolicy)


def train_ppo(
        constants: Dict[str, Any],
        params: Dict[str, Any],
        policy: Type[BasePolicy] = CnnPolicy,
        is_augmented_ppo: bool = False,
):
    is_recurrent_policy = policy in RECURRENT_POLICIES
    is_augmented_policy = policy in AUGMENTED_POLICIES
    # use AugmentedPPO2 to force exploration during policy rollout
    model = AugmentedPPO2 if is_augmented_ppo else PPO2

    env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'visible_training', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots',
    ]
    # detect if extra_feature is used
    use_extra_feature = False
    if 'extra_features' in constants and 'extra_features_norm_factor' in constants \
            and len(constants['extra_features']) > 0 \
            and len(constants['extra_features_norm_factor']) > 0:
        env_kwargs_keys.append('extra_features')
        env_kwargs_keys.append('extra_features_norm_factor')
        use_extra_feature = True
    if use_extra_feature and not is_augmented_policy:
        raise ValueError('use_extra_feature must be used together with an augmented policy')
    # detect if complete_before_timeout_reward is used
    if 'complete_before_timeout_reward' in constants:
        env_kwargs_keys.append('complete_before_timeout_reward')
    env_kwargs = collect_kv(constants, params, keys=env_kwargs_keys)
    env_kwargs['visible'] = env_kwargs['visible_training']
    del env_kwargs['visible_training']
    env_kwargs['is_sync'] = True
    env = make_vec_env(
        DoomEnv, n_envs=params['num_envs'], env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if params['use_multi_threads'] else DummyVecEnv,
    )

    if 'use_curiosity' in params and params['use_curiosity']:
        # default to RDN curiosity wrapper
        if 'curiosity_type' not in params:
            params['curiosity_type'] = 'RDN'

        if params['curiosity_type'] == 'RDN':
            try:
                env = RdnWrapper.load(
                    params['curiosity_load_path'], env,
                    intrinsic_reward_weight=params['intrinsic_reward_weight'],
                    norm_ext_reward=params['normalize_extrinsic_reward'],
                    buffer_size=params['curiosity_buffer_size'],
                    train_freq=params['curiosity_train_freq'],
                    opt_steps=params['curiosity_opt_steps'],
                    batch_size=params['curiosity_batch_size'],
                    gamma=params['curiosity_gamma'],
                    learning_rate=params['curiosity_learning_rate'],
                )
                print("RDN Curiosity model loaded")
            except ValueError:
                print("Failed to load RDN curiosity model, creating new...")
                env = RdnWrapper(
                    env=env,
                    intrinsic_reward_weight=params['intrinsic_reward_weight'],
                    norm_ext_reward=params['normalize_extrinsic_reward'],
                    buffer_size=params['curiosity_buffer_size'],
                    train_freq=params['curiosity_train_freq'],
                    opt_steps=params['curiosity_opt_steps'],
                    batch_size=params['curiosity_batch_size'],
                    gamma=params['curiosity_gamma'],
                    learning_rate=params['curiosity_learning_rate'],
                    drop_last_channel=use_extra_feature,
                )
        elif params['curiosity_type'] == 'ICM':
            try:
                env = IcmWrapper.load(
                    params['curiosity_load_path'], env,
                    intrinsic_reward_weight=params['intrinsic_reward_weight'],
                    norm_ext_reward=params['normalize_extrinsic_reward'],
                    buffer_size=params['curiosity_buffer_size'],
                    train_freq=params['curiosity_train_freq'],
                    opt_steps=params['curiosity_opt_steps'],
                    batch_size=params['curiosity_batch_size'],
                    gamma=params['curiosity_gamma'],
                    learning_rate=params['curiosity_learning_rate'],
                )
                print("ICM Curiosity model loaded")
            except ValueError:
                print("Failed to load ICM curiosity model, creating new...")
                env = IcmWrapper(
                    env=env,
                    intrinsic_reward_weight=params['intrinsic_reward_weight'],
                    norm_ext_reward=params['normalize_extrinsic_reward'],
                    buffer_size=params['curiosity_buffer_size'],
                    train_freq=params['curiosity_train_freq'],
                    opt_steps=params['curiosity_opt_steps'],
                    batch_size=params['curiosity_batch_size'],
                    gamma=params['curiosity_gamma'],
                    learning_rate=params['curiosity_learning_rate'],
                    beta=params['curiosity_icm_beta'],
                    n_hidden=params['curiosity_icm_n_hidden'],
                    drop_last_channel=use_extra_feature,
                )
        else:
            raise ValueError(f'unknown curiosity type specified: {params["curiosity_type"]}')

    lr_schedule = linear_schedule(
        params['learning_rate_beg'], params['learning_rate_end'], verbose=True)
    policy_kwargs = None
    if use_extra_feature:
        policy_kwargs = {
            'n_extra_features': len(constants['extra_features']),
        }
    try:
        agent = model.load(
            params['save_path'], env=env,
            gamma=params['discount_factor'],
            n_steps=params['max_steps_per_episode'],
            ent_coef=params['ent_coef'], vf_coef=params['vf_coef'],
            max_grad_norm=params['grad_clip_norm'],
            noptepochs=params['opt_epochs_per_batch'],
            cliprange=params['ppo_cliprange'],
        )
        agent.learning_rate = lr_schedule
        print("Model loaded")
    except ValueError:
        print("Failed to load model, training from scratch...")
        agent = model(
            policy, env,
            gamma=params['discount_factor'],
            n_steps=params['max_steps_per_episode'],
            ent_coef=params['ent_coef'], vf_coef=params['vf_coef'],
            learning_rate=lr_schedule,
            max_grad_norm=params['grad_clip_norm'],
            noptepochs=params['opt_epochs_per_batch'],
            cliprange=params['ppo_cliprange'],
            cliprange_vf=-1,  # disable value clipping as per original PPO paper
            verbose=True,
            policy_kwargs=policy_kwargs,
        )

    # save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=params['ckpt_freq'], save_path=params['ckpt_path'],
        name_prefix='rl_model', verbose=1,
    )
    # eval periodically
    eval_env_kwargs = env_kwargs
    if 'eval_scenario_cfg_path' in constants:
        eval_env_kwargs['scenario_cfg_path'] = constants['eval_scenario_cfg_path']
    eval_env_kwargs['visible'] = False
    eval_env_kwargs['is_sync'] = True
    eval_env = make_vec_env(
        DoomEnv, n_envs=1, env_kwargs=eval_env_kwargs,
        vec_env_cls=SubprocVecEnv if params['use_multi_threads'] else DummyVecEnv,
    )
    if is_recurrent_policy:
        eval_callback = RecurrentEvalCallback(
            eval_env, n_training_envs=params['num_envs'],
            best_model_save_path=params['best_ckpt_path'],
            log_path=params['log_path'], eval_freq=params['eval_freq'],
            n_eval_episodes=params['num_eval_episodes'],
            deterministic=params['deterministic_eval'], render=False, verbose=1,
        )
    else:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=params['best_ckpt_path'],
            log_path=params['log_path'], eval_freq=params['eval_freq'],
            n_eval_episodes=params['num_eval_episodes'],
            deterministic=params['deterministic_eval'], render=False, verbose=1,
        )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    try:
        agent.learn(total_timesteps=params['total_timesteps'], callback=callbacks)
    finally:
        # save trained agent
        agent.save(save_path=params['save_path'])
        # save curiosity model
        if 'use_curiosity' in params and params['use_curiosity'] and params['curiosity_save_path'] is not None:
            env.save(params['curiosity_save_path'])


def _get_random_obs(constants: Dict[str, Any], params: Dict[str, Any]) -> np.ndarray:
    n_channel = params['history_length']
    if params['use_attention']:
        n_channel *= 2
    if 'extra_features' in constants and 'extra_features_norm_factor' in constants \
            and len(constants['extra_features']) > 0 \
            and len(constants['extra_features_norm_factor']) > 0:
        n_channel += 1
    return np.random.normal(
        size=(constants['resized_height'], constants['resized_width'], n_channel)
    )


def evaluate_ppo(
        constants: Dict[str, Any], params: Dict[str, Any], policy: Type[BasePolicy] = CnnPolicy,
        episodes_to_eval: int = 10, deterministic: bool = False,
        overwrite_frames_to_skip: Optional[int] = None
):
    is_recurrent_policy = policy in RECURRENT_POLICIES
    is_augmented_policy = policy in AUGMENTED_POLICIES

    eval_env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots'
    ]
    # detect if extra_features is used
    use_extra_feature = False
    if 'extra_features' in constants and 'extra_features_norm_factor' in constants \
            and len(constants['extra_features']) > 0 \
            and len(constants['extra_features_norm_factor']) > 0:
        eval_env_kwargs_keys.append('extra_features')
        eval_env_kwargs_keys.append('extra_features_norm_factor')
        use_extra_feature = True
    if use_extra_feature and not is_augmented_policy:
        raise ValueError('use_extra_feature must be used together with an augmented policy')
    # detect if complete_before_timeout_reward is used
    if 'complete_before_timeout_reward' in constants:
        eval_env_kwargs_keys.append('complete_before_timeout_reward')
    eval_env_kwargs = collect_kv(constants, params, keys=eval_env_kwargs_keys)
    if 'eval_scenario_cfg_path' in constants:
        eval_env_kwargs['scenario_cfg_path'] = constants['eval_scenario_cfg_path']
    eval_env_kwargs['visible'] = True
    eval_env_kwargs['is_sync'] = False
    if overwrite_frames_to_skip is not None:
        eval_env_kwargs['frames_to_skip'] = overwrite_frames_to_skip
    eval_env = DoomEnv(**eval_env_kwargs)

    try:
        agent = PPO2.load(params['save_path'])
        print("Model loaded")
    except ValueError:
        print("Failed to load model, evaluate untrained model...")
        agent = PPO2(policy, eval_env, verbose=True)

    # bootstrap the network
    random_obs = _get_random_obs(constants, params)
    if is_recurrent_policy:
        random_zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
        random_zero_completed_obs[0, :] = random_obs
        random_obs = random_zero_completed_obs
    agent.predict(random_obs)

    # evaluation loop
    rewards = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            episode_r = 0.0
            obs = eval_env.reset()
            if is_recurrent_policy:
                state = None
                zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
                zero_completed_obs[0, :] = obs
            done = False
            while not done:
                if is_recurrent_policy:
                    action, state = agent.predict(zero_completed_obs, state, deterministic=deterministic)
                    new_obs, step_r, done, _ = eval_env.step(action[0], smooth_rendering=True)
                    zero_completed_obs[0, :] = new_obs
                else:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    obs, step_r, done, _ = eval_env.step(action, smooth_rendering=True)
                episode_r += step_r
            rewards.append(episode_r)
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_r)

    rewards = np.array(rewards, dtype=np.float32)
    print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')


def record_evaluate_ppo(
        constants: Dict[str, Any], params: Dict[str, Any],
        action_names: List[str],
        filename: str = './evaluation.mp4',
        policy: Type[BasePolicy] = CnnPolicy,
        episodes_to_eval: int = 1, deterministic: bool = False,
        overwrite_frames_to_skip: Optional[int] = None,
        overwrite_episode_timeout: Optional[int] = None,
):
    assert len(action_names) == constants['num_actions'], 'length of action_names and num_actions mismatch'

    is_recurrent_policy = policy in RECURRENT_POLICIES
    is_augmented_policy = policy in AUGMENTED_POLICIES

    eval_env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots'
    ]
    # detect if extra_features is used
    use_extra_feature = False
    if 'extra_features' in constants and 'extra_features_norm_factor' in constants \
            and len(constants['extra_features']) > 0 \
            and len(constants['extra_features_norm_factor']) > 0:
        eval_env_kwargs_keys.append('extra_features')
        eval_env_kwargs_keys.append('extra_features_norm_factor')
        use_extra_feature = True
    if use_extra_feature and not is_augmented_policy:
        raise ValueError('use_extra_feature must be used together with an augmented policy')
    # detect if complete_before_timeout_reward is used
    if 'complete_before_timeout_reward' in constants:
        eval_env_kwargs_keys.append('complete_before_timeout_reward')
    eval_env_kwargs = collect_kv(constants, params, keys=eval_env_kwargs_keys)
    if 'eval_scenario_cfg_path' in constants:
        eval_env_kwargs['scenario_cfg_path'] = constants['eval_scenario_cfg_path']
    eval_env_kwargs['visible'] = True
    eval_env_kwargs['is_sync'] = False
    eval_env_kwargs['screen_format'] = vzd.ScreenFormat.CRCGCB
    if overwrite_frames_to_skip is not None:
        eval_env_kwargs['frames_to_skip'] = overwrite_frames_to_skip
    eval_env = DoomEnv(**eval_env_kwargs, overwrite_episode_timeout=overwrite_episode_timeout)

    try:
        agent = PPO2.load(params['save_path'])
        print("Model loaded")
    except ValueError:
        print("Failed to load model, evaluate untrained model...")
        agent = PPO2(policy, eval_env, verbose=True)

    # bootstrap the network
    random_obs = _get_random_obs(constants, params)
    if is_recurrent_policy:
        random_zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
        random_zero_completed_obs[0, :] = random_obs
        random_obs = random_zero_completed_obs
    agent.predict(random_obs)

    frames = []
    rewards = []
    action_probs = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            episode_r = 0.0
            obs = eval_env.reset()
            if is_recurrent_policy:
                state = None
                zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
                zero_completed_obs[0, :] = obs
            done = False
            while not done:
                if is_recurrent_policy:
                    action, state = agent.predict(zero_completed_obs, state, deterministic=deterministic)
                    action_prob = agent.action_probability(zero_completed_obs, state)
                    action_prob = action_prob[0]
                    new_obs, step_r, done, info = eval_env.step(action[0], smooth_rendering=True)
                    zero_completed_obs[0, :] = new_obs
                else:
                    action, _ = agent.predict(obs, deterministic=deterministic)
                    action_prob = agent.action_probability(obs)
                    obs, step_r, done, info = eval_env.step(action, smooth_rendering=True)
                frames.extend(info['frames'])
                action_probs.extend([action_prob] * len(info['frames']))
                episode_r += step_r
            rewards.append(episode_r)
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_r)
    rewards = np.array(rewards, dtype=np.float32)
    print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')
    eval_env.close()

    def _render_step(f: np.ndarray, p: np.ndarray) -> np.ndarray:
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
        fig.tight_layout(pad=0.2)
        ax[0].imshow(f.transpose(1, 2, 0))  # (ch, h, w) -> (h, w, ch)
        ax[1].bar(action_names, p)
        ax[1].set_ylim(0.0, 1.0)  # use fixed y axis to stabilize animation
        rst_frame = get_img_from_fig(fig, 180, rgb=False)
        plt.close(fig)
        return rst_frame

    # generate prob histograms along with frames and write out frames to video file
    test_frame = _render_step(frames[0], action_probs[0])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h = test_frame.shape[0]
    w = test_frame.shape[1]
    out = cv2.VideoWriter(filename, fourcc, 30, (w, h))
    print("Rendering video...")
    for f, p in tqdm.tqdm(zip(frames, action_probs), total=len(frames)):
        out.write(_render_step(f, p))
    out.release()


def deathmatch_ppo(
        constants: Dict[str, Any], params: Dict[str, Any], policy: Type[BasePolicy] = CnnPolicy,
        deterministic: bool = False,
        overwrite_frames_to_skip: Optional[int] = None
):
    is_recurrent_policy = policy in RECURRENT_POLICIES
    is_augmented_policy = policy in AUGMENTED_POLICIES

    eval_env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots'
    ]
    # detect if extra_features is used
    use_extra_feature = False
    if 'extra_features' in constants and 'extra_features_norm_factor' in constants \
            and len(constants['extra_features']) > 0 \
            and len(constants['extra_features_norm_factor']) > 0:
        eval_env_kwargs_keys.append('extra_features')
        eval_env_kwargs_keys.append('extra_features_norm_factor')
        use_extra_feature = True
    if use_extra_feature and not is_augmented_policy:
        raise ValueError('use_extra_feature must be used together with an augmented policy')
    # detect if complete_before_timeout_reward is used
    if 'complete_before_timeout_reward' in constants:
        eval_env_kwargs_keys.append('complete_before_timeout_reward')
    eval_env_kwargs = collect_kv(constants, params, keys=eval_env_kwargs_keys)
    if 'eval_scenario_cfg_path' in constants:
        eval_env_kwargs['scenario_cfg_path'] = constants['eval_scenario_cfg_path']
    eval_env_kwargs['visible'] = True
    eval_env_kwargs['is_sync'] = False
    if overwrite_frames_to_skip is not None:
        eval_env_kwargs['frames_to_skip'] = overwrite_frames_to_skip

    agent = None
    try:
        agent = PPO2.load(params['save_path'])
        print("Model loaded")
    except ValueError:
        print("Failed to load model")
        exit(-1)

    # bootstrap the network
    random_obs = _get_random_obs(constants, params)
    if is_recurrent_policy:
        # random_zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
        # TODO: use variable observation_space instead of fixed numbers
        random_zero_completed_obs = np.zeros((params['num_envs'], 120, 120, 9))
        random_zero_completed_obs[0, :] = random_obs
        random_obs = random_zero_completed_obs
    agent.predict(random_obs)

    eval_env = DoomEnv(**eval_env_kwargs, overwrite_episode_timeout=0)

    obs = eval_env.reset(new_episode=False)
    if is_recurrent_policy:
        state = None
        zero_completed_obs = np.zeros((params['num_envs'],) + eval_env.observation_space.shape)
        zero_completed_obs[0, :] = obs
    done = False
    frags = 0
    deaths = 0
    i = 0
    while not done:
        i += 1
        frags = int(eval_env.env.get_game_variable(vzd.GameVariable.FRAGCOUNT))
        deaths = int(eval_env.env.get_game_variable(vzd.GameVariable.DEATHCOUNT))
        if i % 300 == 0:
            print(f'FRAGS: {frags}')
            print(f'DEATHS: {deaths}')
        if is_recurrent_policy:
            action, state = agent.predict(zero_completed_obs, state, deterministic=deterministic)
            new_obs, _, done, _ = eval_env.step(action[0], smooth_rendering=True)
            zero_completed_obs[0, :] = new_obs
        else:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, _, done, _ = eval_env.step(action, smooth_rendering=True)

    print(f'Final FRAGS: {frags}')
    print(f'Final DEATHS: {deaths}')
