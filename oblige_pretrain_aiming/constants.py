#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 1/27/21 7:08 PM
# @Desc  : some constant definitions

import sys
import os

import vizdoom as vzd

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from common.utils import make_expand_action_f

# atomic actions
NUM_ATOMIC_ACTIONS = 8
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
TURN_LEFT = 4
TURN_RIGHT = 5
ATTACK = 6
USE = 7

_expand_action = make_expand_action_f(NUM_ATOMIC_ACTIONS)

# action space (both atomic and combination actions)
ACTION_LIST = [
    _expand_action(),

    _expand_action(MOVE_FORWARD),
    _expand_action(MOVE_BACKWARD),
    _expand_action(MOVE_LEFT),
    _expand_action(MOVE_RIGHT),
    _expand_action(TURN_LEFT),
    _expand_action(TURN_RIGHT),
    _expand_action(ATTACK),
    _expand_action(USE),

    # _expand_action(MOVE_FORWARD, TURN_LEFT),
    # _expand_action(MOVE_FORWARD, TURN_RIGHT),
    # _expand_action(MOVE_LEFT, TURN_RIGHT),
    # _expand_action(MOVE_RIGHT, TURN_LEFT),
    #
    # _expand_action(MOVE_FORWARD, ATTACK),
    # _expand_action(MOVE_BACKWARD, ATTACK),
    # _expand_action(MOVE_LEFT, ATTACK),
    # _expand_action(MOVE_RIGHT, ATTACK),
]

CONSTANTS_DICT = {
    'scenario_cfg_path': '../scenarios/oblige_pretrain_aiming.cfg',
    # 'eval_scenario_cfg_path': '../scenarios/oblige_pretrain_aiming.cfg',
    'game_args': '+sv_noautoaim 1 +viz_nocheat 0',
    'num_bots': 0,
    'action_list': ACTION_LIST,
    'num_actions': len(ACTION_LIST),
    'resized_height': 120,
    'resized_width': 120,
    'preprocess_shape': (120, 120),
    'extra_features': [
        vzd.GameVariable.HEALTH, vzd.GameVariable.ARMOR,
        vzd.GameVariable.AMMO0, vzd.GameVariable.AMMO1,
        vzd.GameVariable.AMMO2, vzd.GameVariable.AMMO3,
        vzd.GameVariable.AMMO4, vzd.GameVariable.AMMO5,
        vzd.GameVariable.AMMO6, vzd.GameVariable.AMMO7,
        vzd.GameVariable.AMMO8, vzd.GameVariable.AMMO9,
    ],
    'extra_features_norm_factor': [
        100.0, 200.0,
        300.0, 300.0,
        300.0, 300.0,
        300.0, 300.0,
        300.0, 300.0,
        300.0, 300.0,
    ],
    'complete_before_timeout_reward': 0.0,
}
