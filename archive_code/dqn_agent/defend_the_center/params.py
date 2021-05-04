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
LOAD_FROM = 'saved_model'
SAVE_PATH = 'saved_model'
LOAD_REPLAY_BUFFER = True
SAVE_REPLAY_BUFFER = True
RESET_META_INFO = False

WRITE_TENSORBOARD = False
TENSORBOARD_DIR = 'tf_board/'

# If True, use the prioritized experience replay algorithm, instead of regular experience replay
# This is much more computationally expensive, but will also allow for better results. Implementing
# a binary heap, as recommended in the PER paper, would make this less expensive.
USE_PER = True

# How much the replay buffer should sample based on priorities.
# 0 = complete random samples, 1 = completely aligned with priorities
PRIORITY_SCALE = 0.7

TOTAL_FRAMES = 30_000_000  # Total number of frames to train for
EPS_ANNEALING_FRAMES = 400_000
# MAX_EPISODE_LENGTH = 18000  # Maximum length of an episode (in frames)
FRAMES_BETWEEN_EVAL = 5_000  # Number of frames between evaluations
EVAL_LENGTH = 500  # Number of frames to evaluate for

DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
MEM_SIZE = 10_000  # The maximum size of the replay buffer
MIN_REPLAY_BUFFER_SIZE = 500  # The minimum size the replay buffer must be before we start to update the agent

UPDATE_FREQ = 4  # Number of actions between gradient descent steps
TARGET_UPDATE_FREQ = 1000  # Number of actions between when the target network is updated

# Size of the preprocessed input frame. With the current model architecture, anything below ~80 won't work.
INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 64  # Number of samples the agent learns from at once
HISTORY_LENGTH = 4

FRAMES_TO_SKIP = 2
LEARNING_RATE = 0.00025

VISIBLE_TRAINING = True
USE_REWARD_SHAPING = True
