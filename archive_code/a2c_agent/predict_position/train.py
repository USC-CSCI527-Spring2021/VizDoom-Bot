#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: harry
# @Date  : 2/12/21 5:56 AM
# @Desc  : Train A2C agent

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import tqdm
import numpy as np
import tensorflow as tf

from a2c_common.model import ActorCritic
from a2c_common.agent import A2CAgent
from a2c_common.game_wrapper import GameWrapper
from constants import *
from params import *


def train():
    # create game env
    game = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        (RESIZED_HEIGHT, RESIZED_WIDTH),
        FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=VISIBLE_TRAINING,
        is_sync=True,
        reward_shaper=None,
    )

    # create and build model
    model = ActorCritic(num_actions=NUM_ACTIONS)
    model.build(input_shape=(None, RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    model.summary()

    # create agent and try to load model
    agent = A2CAgent(model, game, NUM_ACTIONS)
    if LOAD_PATH is None or not os.path.exists(LOAD_PATH):
        print('No saved model found, training from scratch')
    else:
        agent.load(LOAD_PATH)
        print(f'Model loaded from {LOAD_PATH}')

    # create optimizer
    # optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

    # train loop
    try:
        with tqdm.trange(TOTAL_EPISODES) as t:
            for i in t:
                _ = game.reset()
                episode_reward = float(agent.train_step_ppo(
                    MAX_STEPS_PER_EPISODE, BATCH_SIZE,
                    optimizer, DISCOUNT_FACTOR,
                    ENTROPY_COFF, CRITIC_COFF,
                    reward_shaping=False,
                    standardize_returns=STANDARDIZE_RETURNS,
                    epochs_per_batch=EPOCHS_PER_BATCH,
                    epsilon=EPSILON,
                ))
                t.set_description(f'Episode {i}')
                t.set_postfix(
                    episode_reward=episode_reward)
    except:
        agent.save(SAVE_PATH)

    agent.save(SAVE_PATH)


if __name__ == '__main__':
    train()
