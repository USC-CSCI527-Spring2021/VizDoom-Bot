#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 1/27/21 7:08 PM
# @Desc  : some constant definitions

import os

from typing import List

SCENARIO_CFG_PATH = "../scenarios/deadly_corridor.cfg"
RAW_DATA_PATH = "./spec_raw_data/"
CHECKPOINT_PATH = './model_ckpt/imitation_{epoch:04d}.ckpt'
# CHECKPOINT_PATH = './model_ckpt/imitation.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)

# atomic actions
NUM_ATOMIC_ACTIONS = 7
MOVE_LEFT = 0
MOVE_RIGHT = 1
ATTACK = 2
MOVE_FORWARD = 3
MOVE_BACKWARD = 4
TURN_LEFT = 5
TURN_RIGHT = 6


def _expand_action(*args) -> List[bool]:
    a = [False] * NUM_ATOMIC_ACTIONS
    for action_id in args:
        if action_id < NUM_ATOMIC_ACTIONS:
            a[action_id] = True
    return a


# action space (both atomic and combination actions)
ACTION_LIST = []
# no action
# ACTION_LIST.append(_expand_action())
# atomic actions
for i in range(NUM_ATOMIC_ACTIONS):
    ACTION_LIST.append(_expand_action(i))
# combination actions
ACTION_LIST.extend([
    _expand_action(MOVE_LEFT, ATTACK),
    _expand_action(MOVE_RIGHT, ATTACK),
    _expand_action(MOVE_FORWARD, ATTACK),
    _expand_action(MOVE_BACKWARD, ATTACK),

    _expand_action(TURN_LEFT, ATTACK),
    _expand_action(TURN_RIGHT, ATTACK),

    _expand_action(MOVE_FORWARD, MOVE_LEFT),
    _expand_action(MOVE_FORWARD, MOVE_RIGHT),
    _expand_action(MOVE_BACKWARD, MOVE_LEFT),
    _expand_action(MOVE_BACKWARD, MOVE_RIGHT),

    _expand_action(MOVE_FORWARD, TURN_LEFT),
    _expand_action(MOVE_FORWARD, TURN_RIGHT),
    _expand_action(MOVE_BACKWARD, TURN_LEFT),
    _expand_action(MOVE_BACKWARD, TURN_RIGHT),

    _expand_action(MOVE_FORWARD, MOVE_LEFT, ATTACK),
    _expand_action(MOVE_FORWARD, MOVE_RIGHT, ATTACK),
    _expand_action(MOVE_BACKWARD, MOVE_LEFT, ATTACK),
    _expand_action(MOVE_BACKWARD, MOVE_RIGHT, ATTACK),

    _expand_action(MOVE_FORWARD, MOVE_LEFT, TURN_LEFT),
    _expand_action(MOVE_FORWARD, MOVE_RIGHT, TURN_LEFT),
    _expand_action(MOVE_BACKWARD, MOVE_LEFT, TURN_LEFT),
    _expand_action(MOVE_BACKWARD, MOVE_RIGHT, TURN_LEFT),

    _expand_action(MOVE_FORWARD, MOVE_LEFT, TURN_RIGHT),
    _expand_action(MOVE_FORWARD, MOVE_RIGHT, TURN_RIGHT),
    _expand_action(MOVE_BACKWARD, MOVE_LEFT, TURN_RIGHT),
    _expand_action(MOVE_BACKWARD, MOVE_RIGHT, TURN_RIGHT),

    _expand_action(MOVE_FORWARD, TURN_LEFT, ATTACK),
    _expand_action(MOVE_FORWARD, TURN_RIGHT, ATTACK),
    _expand_action(MOVE_BACKWARD, TURN_LEFT, ATTACK),
    _expand_action(MOVE_BACKWARD, TURN_RIGHT, ATTACK),
])

NUM_ACTIONS = len(ACTION_LIST)

RESIZED_HEIGHT, RESIZED_WIDTH = 120, 160
