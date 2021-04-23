#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : params.py.py
# @Author: harry
# @Date  : 2/6/21 3:00 AM
# @Desc  : Parameters for training & evaluating

# from reward_shaper import RewardShaper

PARAMS_DICT = {
    # save / load / ckpt
    'save_path': 'saved_model',
    'load_path': 'saved_model',
    'ckpt_path': 'model_ckpt',
    'ckpt_freq': 10_000,
    'best_ckpt_path': 'best_ckpt',
    'log_path': 'logs',
    # training envs
    'num_envs': 4,
    'use_multi_threads': True,  # set to False to disable multi threading of envs
    'visible_training': True,
    # in-training eval related params
    'eval_freq': 5_000,
    'num_eval_episodes': 5,
    'deterministic_eval': False,
    # common params
    'total_timesteps': 1_000_000,
    'frames_to_skip': 3,
    'history_length': 4,
    'use_attention': False,
    'attention_ratio': 0.5,
    'reward_shaper': None,  # set to None to disable reward shaping
    'learning_rate_beg': 0.0001,
    'learning_rate_end': 0.0001,
    'discount_factor': 0.99,  # gamma
    'max_steps_per_episode': 700,
    'grad_clip_norm': 0.5,
    'opt_epochs_per_batch': 4,
    # actor-critic loss related params
    'ent_coef': 0.01,
    'vf_coef': 0.5,
    # ppo related params
    'ppo_cliprange': 0.2,
    # curiosity related params
    'use_curiosity': True,
    'curiosity_type': 'ICM',
    'intrinsic_reward_weight': 100.0,
    'normalize_extrinsic_reward': True,
    'curiosity_save_path': 'curiosity_saved_model',
    'curiosity_load_path': 'curiosity_saved_model',
    'curiosity_buffer_size': 4096,
    'curiosity_train_freq': 2000,
    'curiosity_opt_steps': 4,
    'curiosity_batch_size': 128,
    'curiosity_gamma': 0.99,
    'curiosity_learning_rate': 0.001,
    'curiosity_icm_beta': 0.2,
    'curiosity_icm_n_hidden': 256,
}
