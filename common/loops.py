#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : loops.py
# @Author: harry
# @Date  : 2/25/21 9:01 PM
# @Desc  : Common training and evaluating loops

import tqdm
import numpy as np

from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, BasePolicy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines import PPO2
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from common.game_wrapper import DoomEnv
from common.utils import linear_schedule, collect_kv
from common.evaluate_recurrent_policy import RecurrentEvalCallback
from common.i_reward_shaper import IRewardShaper
from typing import Dict, Tuple, Any, Type


def train_ppo(constants: Dict[str, Any], params: Dict[str, Any], policy: Type[BasePolicy] = CnnPolicy):
    is_recurrent_policy = policy in (CnnLstmPolicy, CnnLnLstmPolicy)

    env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'visible_training', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots'
    ]
    env_kwargs = collect_kv(constants, params, keys=env_kwargs_keys)
    env_kwargs['visible'] = env_kwargs['visible_training']
    del env_kwargs['visible_training']
    env_kwargs['is_sync'] = True
    env = make_vec_env(
        DoomEnv, n_envs=params['num_envs'], env_kwargs=env_kwargs,
        vec_env_cls=SubprocVecEnv if params['use_multi_threads'] else DummyVecEnv,
    )

    lr_schedule = linear_schedule(
        params['learning_rate_beg'], params['learning_rate_end'], verbose=True)
    try:
        agent = PPO2.load(params['save_path'], env=env)
        agent.learning_rate = lr_schedule
        print("Model loaded")
    except ValueError:
        print("Failed to load model, training from scratch...")
        agent = PPO2(
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
        )

    # save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=params['ckpt_freq'], save_path=params['ckpt_path'],
        name_prefix='rl_model', verbose=1,
    )
    # eval periodically
    eval_env_kwargs = env_kwargs
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
        agent.save(save_path=params['save_path'])


def evaluate_ppo(
        constants: Dict[str, Any], params: Dict[str, Any], policy: Type[BasePolicy] = CnnPolicy,
        episodes_to_eval: int = 10, deterministic: bool = False,
):
    is_recurrent_policy = policy in (CnnLstmPolicy, CnnLnLstmPolicy)

    eval_env_kwargs_keys = [
        'scenario_cfg_path', 'game_args', 'action_list', 'preprocess_shape',
        'frames_to_skip', 'history_length', 'use_attention',
        'attention_ratio', 'reward_shaper', 'num_bots'
    ]
    eval_env_kwargs = collect_kv(constants, params, keys=eval_env_kwargs_keys)
    eval_env_kwargs['visible'] = True
    eval_env_kwargs['is_sync'] = False
    eval_env = DoomEnv(**eval_env_kwargs)

    try:
        agent = PPO2.load(params['save_path'])
        print("Model loaded")
    except ValueError:
        print("Failed to load model, evaluate untrained model...")
        agent = PPO2(policy, eval_env, verbose=True)

    # bootstrap the network
    if params['use_attention']:
        random_obs = np.random.normal(
            size=(constants['resized_height'], constants['resized_width'], params['history_length'] * 2)
        )
    else:
        random_obs = np.random.normal(
            size=(constants['resized_height'], constants['resized_width'], params['history_length'])
        )
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
