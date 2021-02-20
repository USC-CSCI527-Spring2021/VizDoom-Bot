#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py.py
# @Author: harry
# @Date  : 2/18/21 9:17 PM
# @Desc  : Evaluate the agent

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tqdm
import numpy as np

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2, A2C
from common.game_wrapper import DoomEnv
from constants import *
from params import *


def evaluate(episodes_to_eval: int = 10, stochastic: bool = True):
    eval_env = DoomEnv(
        scenario_cfg_path=SCENARIO_CFG_PATH,
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=FRAMES_TO_SKIP,
        history_length=HISTORY_LENGTH,
        visible=True,
        is_sync=False,
        use_attention=USE_ATTENTION,
        attention_ratio=ATTENTION_RATIO,
    )

    try:
        agent = PPO2.load(LOAD_PATH)
        print("Model loaded")
    except:
        print("Failed to load model, evaluate untrained model...")
        agent = PPO2(
            CnnPolicy, eval_env, verbose=True,
        )

    # bootstrap the network
    if USE_ATTENTION:
        random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH * 2))
    else:
        random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    agent.predict(random_state)

    # rewards, _ = evaluate_policy(agent, eval_env, episodes_to_eval, not stochastic, return_episode_rewards=True)
    # rewards = np.array(rewards)
    # print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')

    # evaluation loop
    rewards = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            episode_r = 0.0
            state = eval_env.reset()
            done = False
            while not done:
                action, _ = agent.predict(state, deterministic=not stochastic)
                state, step_r, done, _ = eval_env.step(action, smooth_rendering=True)
                episode_r += step_r
            rewards.append(episode_r)
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_r)

    rewards = np.array(rewards, dtype=np.float32)
    print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')


if __name__ == '__main__':
    evaluate(10, True)
