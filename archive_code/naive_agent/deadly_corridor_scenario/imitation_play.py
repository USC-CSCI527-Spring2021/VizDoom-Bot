#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : imitation_play.py.py
# @Author: harry
# @Date  : 1/27/21 11:59 PM
# @Desc  : Use trained imitation model to play Doom!

import vizdoom as vzd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import itertools as it

from constants import *
from imitation_model import build_imitation_model
from typing import List
from gen_play_data import setup_game_for_imitation
from time import sleep


def reflex_step(model: 'tf.keras.Model', state: 'np.array') -> List[bool]:
    """
    Upon receiving a new game state (i.e. video buffer), use
    the model to predict the next action to take.
    :param model: loaded imitation model.
    :param state: an np array indicating video buffer.
    :return: a list of bool indicating the action to take.
    """
    state = np.expand_dims(state, axis=0)
    resized_state = tf.image.resize(state, (RESIZED_HEIGHT, RESIZED_WIDTH))
    logits = model(resized_state)[0, :]
    return ACTION_LIST[np.argmax(logits.numpy())]


def imitation_play(episodes: int):
    num_channel = 1  # gray scale only

    # build model and load weights
    model = build_imitation_model(RESIZED_HEIGHT, RESIZED_WIDTH, num_channel, NUM_ACTIONS)
    model.summary()
    prev_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if prev_ckpt is None:
        raise FileNotFoundError("no checkpoint found! please train the model first")
    else:
        model.load_weights(prev_ckpt).expect_partial()

    # feed in random image to kickstart the model
    rand_img = np.random.randn(480, 640, num_channel)
    _ = reflex_step(model, rand_img)

    # Sets time that will pause the engine after each action (in seconds)
    # Without this everything would go too fast for you to keep track of what's happening.
    sleep_time = 1.0 / vzd.DEFAULT_TICRATE  # = 0.028

    # start the game
    game = setup_game_for_imitation(is_human_play=False)
    for i in range(episodes):
        game.new_episode()
        while not game.is_episode_finished():
            state = game.get_state()
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            screen_buf = np.expand_dims(screen_buf, axis=-1)  # expand channel dim

            a = reflex_step(model, screen_buf)
            r = game.make_action(a)

            # Prints state's game variables and reward.
            print("State #" + str(n))
            print("Game variables:", vars)
            print("Action:", a)
            print("Reward:", r)
            print("=====================")

            if sleep_time > 0:
                sleep(sleep_time)

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()


if __name__ == '__main__':
    imitation_play(episodes=5)
