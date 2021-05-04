#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : gen_play_data.py
# @Author: harry
# @Date  : 1/27/21 3:51 PM
# @Desc  : Generate playing data played by human player, namely you :)

import vizdoom as vzd
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle
import glob
import os
import tensorflow as tf

from typing import Any, List, Tuple, Optional
from constants import *


def setup_game_for_imitation(is_human_play: Optional[bool] = True) -> 'vzd.DoomGame':
    game = vzd.DoomGame()

    # load custom keybindings
    game.set_doom_config_path("../_vizdoom.ini")
    # load and overwrite some configs
    game.load_config(SCENARIO_CFG_PATH)
    if is_human_play:
        game.set_screen_resolution(vzd.ScreenResolution.RES_1280X960)
    else:
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_depth_buffer_enabled(False)
    game.set_labels_buffer_enabled(False)
    game.set_automap_buffer_enabled(False)
    game.set_objects_info_enabled(False)
    game.set_sectors_info_enabled(False)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(True)
    game.set_render_weapon(True)
    game.set_render_decals(True)  # Bullet holes and blood on the walls
    game.set_render_particles(True)
    game.set_render_effects_sprites(True)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(True)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.TURN_LEFT)
    game.add_available_button(vzd.Button.TURN_RIGHT)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.HEALTH)

    # game.set_episode_timeout(200)
    # game.set_episode_start_time(10)
    game.set_window_visible(True)
    if is_human_play:
        game.set_mode(vzd.Mode.SPECTATOR)
    else:
        game.set_mode(vzd.Mode.PLAYER)
    # Enables engine output to console.
    # game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    return game


def do_countdown(sec: int):
    while sec > 0:
        print(sec)
        time.sleep(1)
        sec -= 1


def save_history_data(history: List[Tuple['np.array', List[float], float]], path: str):
    # resize img here
    for i in range(len(history)):
        img = history[i][0]
        if len(img.shape) != 3:
            img = np.expand_dims(img, axis=-1)
        img = tf.image.resize(img, (RESIZED_HEIGHT, RESIZED_WIDTH)).numpy()
        history[i] = (img, history[i][1], history[i][2])

    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError:
            raise RuntimeError("Can't create dir {} for history data".format(path))

    h_list = glob.glob(os.path.join(path, '*.pkl'))
    next_filename = "raw_play_data_{}.pkl".format(len(h_list))
    with open(os.path.join(path, next_filename), 'wb') as f:
        pickle.dump(history, f)


def play_and_record(game: 'vzd.DoomGame', episodes: Optional[int] = 10, wait_sec: Optional[int] = 3) \
        -> List[Tuple['np.array', List[float], float]]:
    history = list()  # list of (screen_buf, action, reward) pairs

    for i in range(episodes):
        print("Episode #" + str(i + 1) + " starts in...")
        do_countdown(wait_sec)

        # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():
            # Gets the state
            state = game.get_state()

            # Which consists of:
            n = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            # depth_buf = state.depth_buffer
            # labels_buf = state.labels_buffer
            # automap_buf = state.automap_buffer
            # labels = state.labels
            # objects = state.objects
            # sectors = state.sectors

            game.advance_action()
            a = game.get_last_action()
            r = game.get_last_reward()

            # Prints state's game variables and reward.
            print("State #" + str(n))
            print("Game variables:", vars)
            print("Last Action:", a)
            print("Last Reward:", r)
            print("=====================")

            # Save history
            history.append((screen_buf, a, r))

        # Check how the episode went.
        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")

    # It will be done automatically anyway but sometimes you need to do it in the middle of the program...
    game.close()

    return history


def main():
    game = setup_game_for_imitation()
    history = play_and_record(game, episodes=1, wait_sec=3)
    save_history_data(history, RAW_DATA_PATH)
    # im = plt.imshow(history[-1][0], cmap='gray')
    # plt.show()
    # print(history[-1][0])
    # print(history[-1][0].shape)


if __name__ == "__main__":
    main()
