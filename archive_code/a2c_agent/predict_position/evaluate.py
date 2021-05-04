#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py.py
# @Author: harry
# @Date  : 2/12/21 5:51 PM
# @Desc  : Evaluate the agent

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tqdm
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from a2c_common.model import ActorCritic
from a2c_common.agent import A2CAgent
from a2c_common.game_wrapper import GameWrapper
from constants import *
from params import *


def evaluate(episodes_to_eval: int = 10, stochastic: bool = True):
    # create game env
    game = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        (RESIZED_HEIGHT, RESIZED_WIDTH),
        FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=True,
        is_sync=False,
        reward_shaper=None,
    )

    # create and build model
    model = ActorCritic(num_actions=NUM_ACTIONS)
    model.build(input_shape=(None, RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    model.summary()

    # create agent and try to load model
    agent = A2CAgent(model, game, NUM_ACTIONS)
    if LOAD_PATH is None or not os.path.exists(LOAD_PATH):
        print('WARNING: No saved model found, evaluating untrained model')
    else:
        agent.load(LOAD_PATH)
        print(f'Model loaded from {LOAD_PATH}')

    # bootstrap the network
    random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    agent.get_action(random_state)

    # evaluation loop
    rewards = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            r = agent.evaluation_step(stochastic)
            rewards.append(r)
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=r)

    rewards = np.array(rewards, dtype=np.float32)
    print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')


if __name__ == '__main__':
    evaluate(10, stochastic=True)
