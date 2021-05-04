#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : params.py.py
# @Author: harry
# @Date  : 2/6/21 3:00 AM
# @Desc  : Parameters for training & evaluating

from constants import RESIZED_HEIGHT, RESIZED_WIDTH

# Loading and saving information.
# If LOAD_FROM is None or the path is invalid, it will train a new agent.
# If SAVE_PATH is None, it will not save the agent
LOAD_PATH = 'saved_model'
SAVE_PATH = 'saved_model'

TOTAL_EPISODES = 100_000
MAX_STEPS_PER_EPISODE = 300

INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 64
HISTORY_LENGTH = 4
DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
ENTROPY_COFF = 0.01
CRITIC_COFF = 0.5
STANDARDIZE_RETURNS = True
EPOCHS_PER_BATCH = 4
EPSILON = 0.2
REWARD_SHAPING = False

FRAMES_TO_SKIP = 4
LEARNING_RATE = 0.00001

VISIBLE_TRAINING = True
