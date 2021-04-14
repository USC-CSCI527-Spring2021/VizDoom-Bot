#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 1/27/21 7:08 PM
# @Desc  : some constant definitions

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from common.utils import make_expand_action_f

# atomic actions
NUM_ATOMIC_ACTIONS = 7
MOVE_FORWARD = 0
MOVE_BACKWARD = 1
MOVE_LEFT = 2
MOVE_RIGHT = 3
TURN_LEFT = 4
TURN_RIGHT = 5
ATTACK = 6

_expand_action = make_expand_action_f(NUM_ATOMIC_ACTIONS)

# action space (both atomic and combination actions)
ACTION_LIST = [
    _expand_action(),

    _expand_action(MOVE_FORWARD),
    # _expand_action(MOVE_BACKWARD),
    _expand_action(MOVE_LEFT),
    _expand_action(MOVE_RIGHT),
    _expand_action(TURN_LEFT),
    _expand_action(TURN_RIGHT),
    _expand_action(ATTACK),

    _expand_action(MOVE_FORWARD, TURN_LEFT),
    _expand_action(MOVE_FORWARD, TURN_RIGHT),
    _expand_action(MOVE_LEFT, TURN_RIGHT),
    _expand_action(MOVE_RIGHT, TURN_LEFT),

    _expand_action(MOVE_FORWARD, ATTACK),
    _expand_action(MOVE_LEFT, ATTACK),
    _expand_action(MOVE_RIGHT, ATTACK),
]

CONSTANTS_DICT = {
    'scenario_cfg_path': '../scenarios/my_cig_01_lv1.cfg',
    'game_args': '-host 1 -deathmatch +sv_forcerespawn 1 +sv_noautoaim 1 '
                 '+sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 0',
    'num_bots': 8,
    'action_list': ACTION_LIST,
    'num_actions': len(ACTION_LIST),
    'resized_height': 120,
    'resized_width': 120,
    'preprocess_shape': (120, 120),
}
