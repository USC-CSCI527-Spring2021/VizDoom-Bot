#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : params.py.py
# @Author: harry
# @Date  : 2/6/21 3:00 AM
# @Desc  : Parameters for training & evaluating

from constants import RESIZED_HEIGHT, RESIZED_WIDTH

# Loading and saving information.
LOAD_PATH = 'saved_model'
SAVE_PATH = 'saved_model'
CKPT_PATH = 'model_ckpt'
CKPT_FREQ = 10_000
BEST_CKPT_PATH = 'best_ckpt'
EVAL_FREQ = 2000
NUM_EVAL_EPISODES = 5
LOG_PATH = 'logs'

TOTAL_TIMESTEPS = 1_000_000
MAX_STEPS_PER_EPISODE = 256

INPUT_SHAPE = (RESIZED_HEIGHT, RESIZED_WIDTH)
BATCH_SIZE = 128
HISTORY_LENGTH = 4
DISCOUNT_FACTOR = 0.99  # Gamma, how much to discount future rewards
ENTROPY_COEF = 0.01
CRITIC_COEF = 0.5
STANDARDIZE_ADV = True
EPOCHS_PER_BATCH = 4
EPSILON = 0.2
REWARD_SHAPING = False
USE_ATTENTION = True
ATTENTION_RATIO = 0.5
GRADS_CLIP_NORM = 0.5

FRAMES_TO_SKIP = 3
LEARNING_RATE_BEG = 0.00005
LEARNING_RATE_END = 0.00001
NUM_ENVS = 4

VISIBLE_TRAINING = True
