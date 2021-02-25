#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper_test.py
# @Author: harry
# @Date  : 2/25/21 1:02 PM
# @Desc  : Description goes here

from common.game_wrapper import DoomEnv


def test():
    game_args = "-host 1 -deathmatch +timelimit 10.0 +sv_forcerespawn 1 +sv_noautoaim 1 " \
                "+sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
    game = DoomEnv(
        scenario_cfg_path='./scenarios/my_cig_01.cfg',
        action_list=[[False]],
        frames_to_skip=3,
        visible=True,
        is_sync=False,
        is_spec=True,
        game_args=game_args,
        game_map='map01',
        num_bots=100,
    )

    game.reset()
    done = False
    while not done:
        _, r, done, info = game.step(action=0, smooth_rendering=True)
        sr = info['shaping_reward'] if 'shaping_reward' in info else None
        print(f'Reward: {r}')
        if sr is not None:
            print(f'Shaping reward: {sr}')


if __name__ == '__main__':
    test()
