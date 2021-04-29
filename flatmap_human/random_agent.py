#!/usr/bin/env python3

import vizdoom as vzd
import random

game = vzd.DoomGame()
game.load_config("../scenarios/flatmap_lv9.cfg")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
name = "SampleRandomAgent"
color = 0
game.add_game_args("+name {} +colorset {}".format(name, color))
game.init()

# Three sample actions: turn left/right and shoot
actions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

# Play until the game (episode) is over.
while not game.is_episode_finished():

    if game.is_player_dead():
        # Use this to respawn immediately after death, new state will be available.
        game.respawn_player()

        # Or observe the game until automatic respawn.
        # game.advance_action();
        # continue;

    # Analyze the state ... or not
    s = game.get_state()

    # Make your action.
    game.make_action(random.choice(actions))

    # Log your frags every ~5 seconds
    if s.number % 175 == 0:
        print("Frags:", game.get_game_variable(vzd.GameVariable.FRAGCOUNT))

game.close()
