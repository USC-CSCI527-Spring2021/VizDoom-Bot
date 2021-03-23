#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : record_evaluate_ppo_lstm.py.py
# @Author: harry
# @Date  : 3/15/21 5:29 PM
# @Desc  : Evaluate and record a video

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, BasePolicy
from common.loops import *
from constants import *
from params import *

if __name__ == '__main__':
    record_evaluate_ppo(
        CONSTANTS_DICT, PARAMS_DICT,
        action_names=['N', 'L', 'R', 'A', 'F', 'TL', 'TR'],
        filename='./evaluation.mp4',
        policy=CnnLstmPolicy,
        episodes_to_eval=1,
        deterministic=False,
        overwrite_frames_to_skip=None,
    )
