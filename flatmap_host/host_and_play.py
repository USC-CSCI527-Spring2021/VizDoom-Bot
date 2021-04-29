#!/usr/bin/env python3

#####################################################################
# This script presents how to host a deathmatch game.
#####################################################################

from __future__ import print_function
from vizdoom import *

N_BOTS = 8

game = DoomGame()

# Use CIG example config or your own.
game.load_config("../scenarios/flatmap_lv9.cfg")
game.set_episode_timeout(0)

game.set_doom_map("map01")  # Limited deathmatch.
# game.set_doom_map("map02")  # Full deathmatch.

# Host game with options that will be used in the competition.
game.add_game_args(
    "-host 2 "
    "-deathmatch "  # Deathmatch rules are used for the game.
    "+timelimit 10.0 "  # The game (episode) will end after this many minutes have elapsed.
    "+sv_forcerespawn 1 "  # Players will respawn automatically after they die.
    "+sv_noautoaim 1 "  # Autoaim is disabled for all players.
    "+sv_respawnprotect 1 "  # Players will be invulnerable for two second after spawning.
    "+sv_spawnfarthest 1 "  # Players will be spawned as far as possible from any other players.
    "+sv_nocrouch 1 "  # Disables crouching.
    "+viz_respawn_delay 3 "  # Sets delay between respanws (in seconds).
    "+viz_nocheat 1"
)

# This can be used to host game without taking part in it (can be simply added as argument of vizdoom executable).
# game.add_game_args("+viz_spectator 1")

# Name your agent and select color
# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue
game.add_game_args("+name Host +colorset 0")

# During the competition, async mode will be forced for all agents.
# game.set_mode(Mode.PLAYER)
game.set_mode(Mode.ASYNC_SPECTATOR)

# game.set_window_visible(False)

game.init()

game.send_game_command("removebots")
for i in range(N_BOTS):
    game.send_game_command("addbot")

frags = 0
deaths = 0
i = 0
while not game.is_episode_finished():
    i += 1
    frags = int(game.get_game_variable(GameVariable.FRAGCOUNT))
    deaths = int(game.get_game_variable(GameVariable.DEATHCOUNT))
    if i % 300 == 0:
        print(f'FRAGS: {frags}')
        print(f'DEATHS: {deaths}')
    game.advance_action()
    if game.is_player_dead():
        game.respawn_player()

print(f'Final FRAGS: {frags}')
print(f'Final DEATHS: {deaths}')
game.close()
