#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : evaluate.py.py
# @Author: harry
# @Date  : 2/6/21 3:02 AM
# @Desc  : Evaluate the model by playing some episodes of Doom

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import tensorflow as tf

from dqn_common.game_wrapper import GameWrapper
from dqn_common.agent import DQNAgent
from model import build_q_network
from constants import *
from params import *


def evaluate(num_episodes=10):
    # create environment
    game_wrapper = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        INPUT_SHAPE, FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=True, is_sync=False
    )

    # build main network
    main_dqn = build_q_network(NUM_ACTIONS, LEARNING_RATE, INPUT_SHAPE, HISTORY_LENGTH)

    # load saved model
    agent = DQNAgent(
        main_dqn, None, None, NUM_ACTIONS,
        INPUT_SHAPE, BATCH_SIZE, HISTORY_LENGTH,
        eps_annealing_frames=EPS_ANNEALING_FRAMES,
        eps_evaluation=0.0,
        replay_buffer_start_size=MEM_SIZE / 2,
        max_frames=TOTAL_FRAMES,
        use_per=USE_PER,
    )
    if os.path.exists(LOAD_FROM) and os.path.isdir(LOAD_FROM):
        print("Loading from", LOAD_FROM)
        _ = agent.load(LOAD_FROM, load_replay_buffer=False)
        print("Loaded")
    else:
        print(f"Saved model path {LOAD_FROM} is invalid")
        exit(-1)

    # bootstrap the network
    random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    _ = agent.get_action(0, random_state, evaluation=True)

    # stats
    rewards = []

    # evaluation loop
    for i in range(num_episodes):
        print("Episode", i + 1)
        game_wrapper.reset()
        episode_reward = 0.0
        terminal = False
        while not terminal:
            # Get action
            action = agent.get_action(0, game_wrapper.state, evaluation=True)

            # Take step
            # XXX: we ignore shaping reward during evaluation
            _, r, terminal, _ = game_wrapper.step(action, smooth_rendering=True)
            episode_reward += r
        print("Reward:", episode_reward)
        rewards.append(episode_reward)

    game_wrapper.stop()
    rewards = np.array(rewards, dtype=float)
    print("===============")
    print("Reward avg:", rewards.mean())
    print("Reward std:", rewards.std())
    print("Reward min:", rewards.min())
    print("Reward max:", rewards.max())


if __name__ == '__main__':
    evaluate(10)
