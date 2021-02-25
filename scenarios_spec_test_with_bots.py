#!/usr/bin/env python3

#####################################################################
# This script presents how to run some scenarios.
# Configuration is loaded from "../../scenarios/<SCENARIO_NAME>.cfg" file.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
#
# To see the scenario description go to "../../scenarios/README.md"
#####################################################################

from __future__ import print_function

import itertools as it
from random import choice
from time import sleep
import vizdoom as vzd
from argparse import ArgumentParser

DEFAULT_CONFIG = "./scenarios/cig.cfg"
if __name__ == "__main__":

    parser = ArgumentParser("ViZDoom scenarios example.")
    parser.add_argument(dest="config",
                        default=DEFAULT_CONFIG,
                        nargs="?",
                        help="Path to the configuration file of the scenario."
                             " Please see "
                             "./scenarios/*.cfg for more scenarios.")

    args = parser.parse_args()
    game = vzd.DoomGame()

    # load custom keybindings
    game.set_doom_config_path("./_vizdoom.ini")
    # Choose scenario config file you wish to watch.
    # Don't load two configs cause the second will overwrite the first one.
    # Multiple config files are ok but combining these ones doesn't make much sense.
    game.load_config(args.config)
    game.set_doom_map("map01")  # Limited deathmatch.
    # game.set_doom_map("map02")  # Full deathmatch.
    # Enables freelook in engine
    game.add_game_args("+freelook 1")
    # Host game with options that will be used in the competition.
    game.add_game_args(
        "-host 1 "
        # This machine will function as a host for a multiplayer game with this many players (including this machine). 
        # It will wait for other machines to connect using the -join parameter and then start the game when everyone is connected.
        "-deathmatch "  # Deathmatch rules are used for the game.
        "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
        "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
        "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
        "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
        "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
        "+sv_nocrouch 1 "  # Disables crouching.
        "+viz_respawn_delay 0 "  # Sets delay between respanws (in seconds).
        "+viz_nocheat 1")  # Disables depth and labels buffer and the ability to use commands that could interfere with multiplayer game.

    # Name your agent and select color
    # colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
    game.add_game_args("+name AI +colorset 0")


    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    # game.set_render_hud(False)
    # game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(True)
    # game.set_render_weapon(True)
    # game.set_render_decals(False)  # Bullet holes and blood on the walls
    # game.set_render_particles(False)
    # game.set_render_effects_sprites(False)  # Smoke and blood
    # game.set_render_messages(False)  # In-game messages
    # game.set_render_corpses(False)
    # game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    game.add_available_game_variable(vzd.HITCOUNT)
    game.add_available_game_variable(vzd.HITS_TAKEN)

    # Makes the screen bigger to see more details.
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_window_visible(True)
    game.set_mode(vzd.Mode.SPECTATOR)
    game.init()

    # Creates all possible actions depending on how many buttons there are.
    actions_num = game.get_available_buttons_size()
    actions = []
    for perm in it.product([False, True], repeat=actions_num):
        actions.append(list(perm))

    episodes = 1
    sleep_time = 0.028
    bots = 4

    for i in range(episodes):
        print("Episode #" + str(i + 1))

        # Add specific number of bots
        # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
        # edit this file to adjust bots).
        game.send_game_command("removebots")
        for i in range(bots):
            game.send_game_command("addbot")

        # Not needed for the first episode but the loop is nicer.
        game.new_episode()
        while not game.is_episode_finished():

            # Gets the state and possibly to something with it
            state = game.get_state()

            game.advance_action()
            a = game.get_last_action()
            r = game.get_last_reward()

            print("State #" + str(state.number))
            print("Game Variables:", state.game_variables)
            print("Last Action:", a)
            print("Last Reward:", r)
            print("=====================")

            # Sleep some time because processing is too fast to watch.
            if sleep_time > 0:
                sleep(sleep_time)

        print("Episode finished!")
        print("total reward:", game.get_total_reward())
        print("************************")
