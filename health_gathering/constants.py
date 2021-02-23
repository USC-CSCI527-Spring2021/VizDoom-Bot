#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 1/27/21 7:08 PM
# @Desc  : some constant definitions

import itertools as it

from typing import List

SCENARIO_CFG_PATH = "../scenarios/health_gathering.cfg"

# atomic actions
NUM_ATOMIC_ACTIONS = 3
TURN_LEFT = 0
TURN_RIGHT = 1
MOVE_FORWARD = 2


def _expand_action(*args) -> List[bool]:
    a = [False] * NUM_ATOMIC_ACTIONS
    for action_id in args:
        if action_id < NUM_ATOMIC_ACTIONS:
            a[action_id] = True
    return a


# action space (both atomic and combination actions)
ACTION_LIST = [
    _expand_action(),
    _expand_action(TURN_LEFT),
    _expand_action(TURN_RIGHT),
    _expand_action(MOVE_FORWARD),
]
NUM_ACTIONS = len(ACTION_LIST)

RESIZED_HEIGHT, RESIZED_WIDTH = 120, 120
