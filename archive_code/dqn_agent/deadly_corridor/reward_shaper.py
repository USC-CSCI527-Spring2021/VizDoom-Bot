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

from dqn_common.i_reward_shaper import IRewardShaper
from dqn_common.game_wrapper import GameWrapper
from constants import *
from typing import List


class RewardShaper(IRewardShaper):
    def __init__(self):
        super(RewardShaper).__init__()
        self.available_vars = [
            vzd.GameVariable.DAMAGE_TAKEN,
            vzd.GameVariable.KILLCOUNT,
        ]
        self.damage_taken = 0.0
        self.kill_count = 0.0

    def update_vars(self, game_vars: 'np.array'):
        self.damage_taken = game_vars[0]
        self.kill_count = game_vars[1]

    def get_subscribed_game_var_list(self) -> List[int]:
        return self.available_vars

    def reset(self, game_vars: 'np.array'):
        assert len(self.available_vars) == game_vars.shape[0]
        self.update_vars(game_vars)

    def calc_reward(self, new_game_vars: 'np.array') -> float:
        assert len(self.available_vars) == new_game_vars.shape[0]

        d_damage_taken = new_game_vars[0] - self.damage_taken
        d_kill_count = new_game_vars[1] - self.kill_count

        self.update_vars(new_game_vars)
        return -0.25 * d_damage_taken + 20.0 * d_kill_count


def test_reward_shaper():
    rs = RewardShaper()
    g = GameWrapper(
        visible=True, is_sync=False,
        scenario_cfg_path="../scenarios/deadly_corridor.cfg",
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=4,
        reward_shaper=rs,
    )
    g.reset()
    print("state shape: ", g.state.shape)

    for e in range(2):
        print("Episode", e)
        g.reset()
        print("Initial damage taken:", rs.damage_taken)
        print("Initial kill count:", rs.kill_count)
        for _ in range(100):
            s, r, t, sr = g.step(np.random.randint(0, len(g.action_list)), smooth_rendering=True)
            print(s.shape, r, t, sr)
            if t:
                break
        print("End of episode {}".format(e))
        print("Final damage taken:", rs.damage_taken)
        print("Final kill count:", rs.kill_count)


if __name__ == '__main__':
    test_reward_shaper()
