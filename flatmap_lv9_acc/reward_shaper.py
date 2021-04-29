#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : reward_shaper.py
# @Author: harry
# @Date  : 2/7/21 5:58 PM
# @Desc  : Reward shaper

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
            vzd.GameVariable.AMMO5,
            vzd.GameVariable.HITCOUNT,
            vzd.GameVariable.KILLCOUNT,
            vzd.GameVariable.HITS_TAKEN,
            vzd.GameVariable.DEAD,
            vzd.GameVariable.ARMOR,
            vzd.GameVariable.POSITION_X,
            vzd.GameVariable.POSITION_Y,
            vzd.GameVariable.POSITION_Z,
            vzd.GameVariable.ANGLE,
            vzd.GameVariable.FRAGCOUNT,
        ]
        self.health = 0.0
        self.ammo5 = 0.0
        self.hit_count = 0.0
        self.kill_count = 0.0
        self.hits_taken = 0.0
        self.dead = 0.0
        self.armor = 0.0
        self.position_x = 0.0
        self.position_y = 0.0
        self.position_z = 0.0
        self.angle = 0.0
        self.frag_count = 0.0

    def update_vars(self, game_vars: 'np.array'):
        self.health = game_vars[0]
        self.ammo5 = game_vars[1]
        self.hit_count = game_vars[2]
        self.kill_count = game_vars[3]
        self.hits_taken = game_vars[4]
        self.dead = game_vars[5]
        self.armor = game_vars[6]
        self.position_x = game_vars[7]
        self.position_y = game_vars[8]
        self.position_z = game_vars[9]
        self.angle = game_vars[10]
        self.frag_count = game_vars[11]
        # print(f'position_x: {self.position_x}')
        # print(f'position_y: {self.position_y}')
        # print(f'position_z: {self.position_z}')
        # print(f'angle: {self.angle}')

    def calc_dist(self, new_x: float, new_y: float, new_z: float) -> float:
        pos = np.array([
            new_x - self.position_x, new_y - self.position_y, new_z - self.position_z],
            dtype=np.float32)
        return float(np.linalg.norm(pos, ord=2))

    def get_subscribed_game_var_list(self) -> List[int]:
        return self.available_vars

    def reset(self, game_vars: 'np.array'):
        assert len(self.available_vars) == game_vars.shape[0]
        self.update_vars(game_vars)

    def calc_reward(self, new_game_vars: 'np.array') -> float:
        assert len(self.available_vars) == new_game_vars.shape[0]

        r = 0.0
        is_dead = (new_game_vars[5] == 1.0) or (self.dead == 1.0)

        if not is_dead:
            d_health = new_game_vars[0] - self.health
            # reward for health-pickups
            r += 0.04 * d_health if d_health > 0 else 0.0
            # penalty for health loss
            r += 0.05 * d_health if d_health < 0 else 0.0
            # extra reward for picking up health in need
            r += 0.08 * d_health if d_health > 0 and self.health <= 10.0 else 0.0

            d_ammo5 = new_game_vars[1] - self.ammo5
            # reward for ammo-pickups
            r += 0.15 * d_ammo5 if d_ammo5 > 0 else 0.0
            # penalty for ammo decrement
            r += 0.10 * d_ammo5 if d_ammo5 < 0 else 0.0
            # extra reward for picking up ammo in need
            r += 0.30 * d_ammo5 if d_ammo5 > 0 and self.ammo5 <= 3 else 0.0

            # reward for hits
            r += 0.5 if new_game_vars[2] > self.hit_count else 0.0
            # reward for kills
            r += 1.0 if new_game_vars[11] > self.frag_count else 0.0
            # penalty for hits taken
            # r += -0.1 if new_game_vars[4] > self.hits_taken else 0.0
            # extra penalty for suicide
            r += -0.5 if new_game_vars[11] < self.frag_count else 0.0
            # reward for armor-pickups
            r += 0.30 if new_game_vars[6] > self.armor else 0.0

            dist = self.calc_dist(new_game_vars[7], new_game_vars[8], new_game_vars[9])
            # reward for moving around
            r += 9e-5 * dist
            # penalty for staying
            r += -0.03 if dist < 8.0 else 0.0

        self.update_vars(new_game_vars)
        return r


def test_reward_shaper():
    game_args = CONSTANTS_DICT['game_args']
    g = DoomEnv(
        scenario_cfg_path=CONSTANTS_DICT['scenario_cfg_path'],
        action_list=ACTION_LIST,
        preprocess_shape=CONSTANTS_DICT['preprocess_shape'],
        frames_to_skip=3,
        history_length=4,
        visible=True,
        is_sync=False,
        is_spec=True,
        use_attention=True,
        attention_ratio=0.5,
        reward_shaper=RewardShaper,
        game_args=game_args,
        num_bots=4,
        overwrite_episode_timeout=None,
        extra_features=[vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO5, vzd.GameVariable.ARMOR],
        extra_features_norm_factor=[100.0, 50.0, 200.0],
    )
    rs = g.reward_shaper

    for e in range(2):
        print("Episode", e)
        g.reset()
        print("Initial ammo5:", rs.ammo5)
        print("Initial frag count:", rs.frag_count)
        # print("Initial kill count:", rs.kill_count)
        t = False
        while not t:
            s, r, t, info = g.step(np.random.randint(0, len(g.action_list)), smooth_rendering=True)
            # print(s.shape)
            # print(s.dtype)
            # print(s[0, 0:10, -1])
            print(s.shape, r, t, info['shaping_reward'])
            if t:
                break
        print("End of episode {}".format(e))
        print("Final ammo5:", rs.ammo5)
        print("Final hit_count:", rs.hit_count)
        print("Final kill_count:", rs.kill_count)
        print("Final hits_taken:", rs.hits_taken)
        print("Final armor:", rs.armor)
        print("Final frag count:", rs.frag_count)


if __name__ == '__main__':
    test_reward_shaper()
