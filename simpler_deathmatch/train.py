#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: harry
# @Date  : 2/18/21 7:30 PM
# @Desc  : Train the agent

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from stable_baselines.common import make_vec_env
from stable_baselines.common.policies import CnnPolicy
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines import PPO2
from common.game_wrapper import DoomEnv
from common.utils import linear_schedule
from constants import *
from params import *
from reward_shaper import RewardShaper


def train():
    env_kwargs = {
        'scenario_cfg_path': SCENARIO_CFG_PATH,
        'action_list': ACTION_LIST,
        'preprocess_shape': (RESIZED_HEIGHT, RESIZED_WIDTH),
        'frames_to_skip': FRAMES_TO_SKIP,
        'history_length': HISTORY_LENGTH,
        'visible': VISIBLE_TRAINING,
        'use_attention': USE_ATTENTION,
        'attention_ratio': ATTENTION_RATIO,
        'reward_shaper': RewardShaper if REWARD_SHAPING else None,
    }
    env = make_vec_env(DoomEnv, n_envs=NUM_ENVS, env_kwargs=env_kwargs)

    try:
        agent = PPO2.load(SAVE_PATH, env=env)
        agent.learning_rate = linear_schedule(LEARNING_RATE_BEG, LEARNING_RATE_END, verbose=True)
        print("Model loaded")
    except ValueError:
        print("Failed to load model, training from scratch...")
        agent = PPO2(
            CnnPolicy, env,
            gamma=DISCOUNT_FACTOR,
            n_steps=MAX_STEPS_PER_EPISODE,
            ent_coef=ENTROPY_COEF, vf_coef=CRITIC_COEF,
            learning_rate=linear_schedule(LEARNING_RATE_BEG, LEARNING_RATE_END, verbose=True),
            max_grad_norm=GRADS_CLIP_NORM,
            noptepochs=EPOCHS_PER_BATCH,
            cliprange=EPSILON,
            verbose=True,
        )

    # Save a checkpoint periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=CKPT_FREQ, save_path=CKPT_PATH,
        name_prefix='rl_model', verbose=1,
    )
    # eval periodically
    eval_env = DoomEnv(
        scenario_cfg_path=SCENARIO_CFG_PATH,
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=FRAMES_TO_SKIP,
        history_length=HISTORY_LENGTH,
        visible=False,
        is_sync=True,
        use_attention=USE_ATTENTION,
        attention_ratio=ATTENTION_RATIO,
    )
    eval_callback = EvalCallback(
        eval_env, best_model_save_path=BEST_CKPT_PATH,
        log_path=LOG_PATH, eval_freq=EVAL_FREQ, n_eval_episodes=NUM_EVAL_EPISODES,
        deterministic=False, render=False, verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    try:
        agent.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    finally:
        agent.save(save_path=SAVE_PATH)


if __name__ == '__main__':
    train()
