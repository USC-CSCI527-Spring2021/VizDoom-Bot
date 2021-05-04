#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : reward_shaper.py
# @Author: harry
# @Date  : 2/7/21 5:58 PM
# @Desc  : Reward shaper for deadly corridor

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import vizdoom as vzd

from a2c_common.i_reward_shaper import IRewardShaper
from a2c_common.game_wrapper import GameWrapper
from constants import *
from typing import List


class RewardShaper(IRewardShaper):
    def __init__(self):
        super(RewardShaper).__init__()
        self.available_vars = [
            vzd.GameVariable.HEALTH,
        ]
        self.health = 0.0

    def update_vars(self, game_vars: 'np.array'):
        self.health = game_vars[0]

    def get_subscribed_game_var_list(self) -> List[int]:
        return self.available_vars

    def reset(self, game_vars: 'np.array'):
        assert len(self.available_vars) == game_vars.shape[0]
        self.update_vars(game_vars)

    def calc_reward(self, new_game_vars: 'np.array') -> float:
        assert len(self.available_vars) == new_game_vars.shape[0]

        d_health = new_game_vars[0] - self.health

        self.update_vars(new_game_vars)
        return d_health * 0.1


def test_reward_shaper():
    rs = RewardShaper()
    g = GameWrapper(
        visible=True, is_sync=False,
        scenario_cfg_path=SCENARIO_CFG_PATH,
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=3,
        reward_shaper=rs,
    )
    g.reset()
    print("state shape: ", g.state.shape)

    for e in range(2):
        print("Episode", e)
        g.reset()
        print("Initial health:", rs.health)
        for _ in range(200):
            s, r, t, sr = g.step(np.random.randint(0, len(g.action_list)), smooth_rendering=True)
            print(s.shape, r, t, sr)
            if t:
                break
        print("End of episode {}".format(e))
        print("Final health:", rs.health)


if __name__ == '__main__':
    test_reward_shaper()
