#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper_test.py
# @Author: harry
# @Date  : 2/25/21 1:02 PM
# @Desc  : Description goes here

import random
import vizdoom as vzd
import matplotlib.pyplot as plt
import numpy as np

from stable_baselines.common.env_checker import check_env
from common.game_wrapper import DoomEnv
from common.utils import make_expand_action_f


def plot_obs(obs: np.ndarray):
    assert obs.shape[-1] >= 8  # 4 frames + 4 attention frames + additional features
    frames = obs[:, :, :4]
    att_frames = obs[:, :, 4:8]

    fig, axs = plt.subplots(2, 4)
    for i in range(4):
        axs[0, i].imshow(frames[:, :, i], cmap='gray')
        axs[1, i].imshow(att_frames[:, :, i], cmap='gray')

    plt.show()


def test():
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
        _expand_action(MOVE_BACKWARD),
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
        _expand_action(MOVE_BACKWARD, ATTACK),
        _expand_action(MOVE_LEFT, ATTACK),
        _expand_action(MOVE_RIGHT, ATTACK),
    ]
    game_args = "-host 1 -deathmatch +timelimit 10.0 +sv_forcerespawn 1 +sv_noautoaim 1 " \
                "+sv_respawnprotect 1 +sv_spawnfarthest 1 +sv_nocrouch 1 +viz_respawn_delay 0 +viz_nocheat 1"
    env = DoomEnv(
        scenario_cfg_path='./scenarios/flatmap_lv1.cfg',
        action_list=ACTION_LIST,
        preprocess_shape=(120, 120),
        frames_to_skip=3,
        history_length=4,
        visible=True,
        is_sync=False,
        is_spec=False,
        use_attention=True,
        attention_ratio=0.65,
        reward_shaper=None,
        game_args=game_args,
        num_bots=8,
        overwrite_episode_timeout=None,
        extra_features=[vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO5, vzd.GameVariable.ARMOR],
        extra_features_norm_factor=[100.0, 50.0, 200.0],
    )

    # check_env(env)

    obs = env.reset()
    # plot_obs(obs)

    done = False
    i = 0
    while not done and i < 180:
        i += 1
        obs, r, done, info = env.step(action=random.randint(0, len(ACTION_LIST) - 1), smooth_rendering=True)
        print(obs.shape)

    plot_obs(obs)


if __name__ == '__main__':
    test()
