#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train_ppo_lstm.py
# @Author: harry
# @Date  : 2/18/21 7:30 PM
# @Desc  : Train the agent (PPO with a CNN-LSTM policy network)

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

from stable_baselines.common.policies import CnnPolicy, CnnLstmPolicy, CnnLnLstmPolicy, BasePolicy
from common.policies import AugmentedCnnLstmPolicy
from common.loops import *
from constants import *
from params import *

if __name__ == '__main__':
    train_ppo(CONSTANTS_DICT, PARAMS_DICT, policy=AugmentedCnnLstmPolicy, is_augmented_ppo=False)
