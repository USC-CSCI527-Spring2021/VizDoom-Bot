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

from common.i_reward_shaper import IRewardShaper
from common.game_wrapper import DoomEnv
from constants import *
from typing import List


class RewardShaper(IRewardShaper):
    def __init__(self):
        super(RewardShaper).__init__()
        self.available_vars = [
            vzd.GameVariable.HEALTH,
            vzd.GameVariable.AMMO2,
            vzd.GameVariable.KILLCOUNT,
        ]
        self.health = 0.0
        self.ammo2 = 0.0
        self.kill_count = 0.0

    def update_vars(self, game_vars: 'np.array'):
        self.health = game_vars[0]
        self.ammo2 = game_vars[1]
        self.kill_count = game_vars[2]

    def get_subscribed_game_var_list(self) -> List[int]:
        return self.available_vars

    def reset(self, game_vars: 'np.array'):
        assert len(self.available_vars) == game_vars.shape[0]
        self.update_vars(game_vars)

    def calc_reward(self, new_game_vars: 'np.array') -> float:
        assert len(self.available_vars) == new_game_vars.shape[0]

        d_health = new_game_vars[0] - self.health
        d_ammo2 = new_game_vars[1] - self.ammo2
        d_kill_count = new_game_vars[2] - self.kill_count

        self.update_vars(new_game_vars)
        return 2.0 * d_health + 100.0 * d_kill_count


def test_reward_shaper():
    g = DoomEnv(
        scenario_cfg_path=SCENARIO_CFG_PATH,
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=3,
        history_length=4,
        visible=True,
        is_sync=False,
        use_attention=True,
        attention_ratio=0.5,
        reward_shaper=RewardShaper,
    )
    rs = g.reward_shaper

    for e in range(2):
        print("Episode", e)
        g.reset()
        print("Initial ammo2:", rs.ammo2)
        print("Initial kill count:", rs.kill_count)
        for _ in range(200):
            s, r, t, info = g.step(np.random.randint(0, len(g.action_list)), smooth_rendering=True)
            print(s.shape, r, t, info['shaping_reward'])
            if t:
                break
        print("End of episode {}".format(e))
        print("Final ammo2:", rs.ammo2)
        print("Final kill count:", rs.kill_count)


if __name__ == '__main__':
    test_reward_shaper()
