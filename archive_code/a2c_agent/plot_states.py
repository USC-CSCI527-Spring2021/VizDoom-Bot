#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot_states.py
# @Author: harry
# @Date  : 2/14/21 3:26 PM
# @Desc  : Plot game states

import matplotlib.pyplot as plt

from a2c_common.game_wrapper import GameWrapper


def plot_states():
    def _expand_action(*args):
        a = [False] * NUM_ATOMIC_ACTIONS
        for action_id in args:
            if action_id < NUM_ATOMIC_ACTIONS:
                a[action_id] = True
        return a

    NUM_ATOMIC_ACTIONS = 3
    TURN_lEFT = 0
    TURN_RIGHT = 1
    ATTACK = 2
    ACTION_LIST = [
        _expand_action(),
        _expand_action(TURN_lEFT),
        _expand_action(TURN_RIGHT),
        _expand_action(ATTACK),
    ]

    # create game env
    game = GameWrapper(
        "./scenarios/defend_the_line.cfg", ACTION_LIST,
        (120, 120),
        frames_to_skip=4, history_length=4,
        visible=True,
        is_sync=False,
        reward_shaper=None,
    )
    game.reset()

    for _ in range(4):
        game.step(1)

    s = game.get_state()

    r, c = 2, 2
    fig, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            ax[i, j].imshow(s[:, :, i * c + j], cmap='gray')

    plt.show()

    game.env.close()


def plot_states_with_attention():
    def _expand_action(*args):
        a = [False] * NUM_ATOMIC_ACTIONS
        for action_id in args:
            if action_id < NUM_ATOMIC_ACTIONS:
                a[action_id] = True
        return a

    NUM_ATOMIC_ACTIONS = 3
    TURN_lEFT = 0
    TURN_RIGHT = 1
    ATTACK = 2
    ACTION_LIST = [
        _expand_action(),
        _expand_action(TURN_lEFT),
        _expand_action(TURN_RIGHT),
        _expand_action(ATTACK),
    ]

    # create game env
    game = GameWrapper(
        "./scenarios/predict_position.cfg", ACTION_LIST,
        (120, 120),
        frames_to_skip=4, history_length=4,
        visible=True,
        is_sync=False,
        reward_shaper=None,
        use_attention=True,
        attention_ratio=0.5,
    )
    game.reset()

    for _ in range(4):
        game.step(1)

    s = game.get_state()

    r, c = 2, 4
    fig, ax = plt.subplots(r, c)
    for i in range(r):
        for j in range(c):
            ax[i, j].imshow(s[:, :, i * c + j], cmap='gray')

    plt.show()

    game.env.close()


if __name__ == '__main__':
    plot_states()
    plot_states_with_attention()
