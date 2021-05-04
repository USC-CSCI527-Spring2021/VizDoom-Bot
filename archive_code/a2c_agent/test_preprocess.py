#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : plot_states.py
# @Author: harry
# @Date  : 2/14/21 3:26 PM
# @Desc  : Test process_frame

import matplotlib.pyplot as plt

from a2c_common.utils import process_frame
from a2c_common.game_wrapper import GameWrapper


def test_preprocess():
    NUM_ATOMIC_ACTIONS = 3
    TURN_lEFT = 0
    TURN_RIGHT = 1
    ATTACK = 2

    def _expand_action(*args):
        a = [False] * NUM_ATOMIC_ACTIONS
        for action_id in args:
            if action_id < NUM_ATOMIC_ACTIONS:
                a[action_id] = True
        return a

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

    # take one frame
    frame = game.frame

    frame_resized = process_frame(frame, (120, 120), normalize=True, zoom_in=False)
    frame_zoomed = process_frame(frame, (120, 120), normalize=True, zoom_in=True, zoom_in_ratio=0.5)

    fig, ax = plt.subplots(2)
    ax[0].imshow(frame_resized, cmap='gray')
    ax[1].imshow(frame_zoomed, cmap='gray')
    plt.show()

    game.env.close()


if __name__ == '__main__':
    test_preprocess()
