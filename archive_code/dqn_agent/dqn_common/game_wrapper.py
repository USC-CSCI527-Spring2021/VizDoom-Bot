#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper.py
# @Author: harry
# @Date  : 2/4/21 7:15 PM
# @Desc  : A wrapper class for VizDoom game

import time
import vizdoom as vzd
import numpy as np

from dqn_common.utils import process_frame
from dqn_common.i_reward_shaper import IRewardShaper
from typing import List


class GameWrapper:
    def __init__(
            self,
            scenario_cfg_path: str,
            action_list: List[List[bool]],
            preprocess_shape=(60, 80),
            frames_to_skip=4,
            history_length=4,
            visible=False,
            is_sync=True,
            reward_shaper: 'IRewardShaper' = None
    ):
        game = vzd.DoomGame()
        game.load_config(scenario_cfg_path)
        game.set_window_visible(visible)
        if is_sync:
            game.set_mode(vzd.Mode.PLAYER)
        else:
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
        game.set_screen_format(vzd.ScreenFormat.GRAY8)
        game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
        if reward_shaper is not None:
            game.set_available_game_variables(reward_shaper.get_subscribed_game_var_list())
        game.init()
        self.env = game
        self.action_list = action_list
        self.preprocess_shape = preprocess_shape
        self.frames_to_skip = frames_to_skip
        self.history_length = history_length
        self.reward_shaper = reward_shaper

        self.state = None
        self.frame = None

    def reset(self):
        """
        Resets the environment
        """
        self.env.new_episode()
        init_state = self.env.get_state()
        self.frame = init_state.screen_buffer
        if self.reward_shaper is not None:
            self.reward_shaper.reset(init_state.game_variables)

        # For the initial state, we stack the first frame history_length times
        self.state = np.repeat(process_frame(self.frame, self.preprocess_shape), self.history_length, axis=-1)

    def step(self, action, smooth_rendering=False):
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            smooth_rendering: Whether render intermediate states to make game looks smoother;
                skip rendering tics could potentially expedite training
        Returns:
            processed_frame: The processed new frame as a result of that action
            reward: The reward for taking that action
            terminal: Whether the game has ended
            shaping_reward: Optional shaping reward for training. Applicable only if
                self.reward_shaper is not None
        """
        if not smooth_rendering:
            # make_action will not update(render) skipped tics
            reward = self.env.make_action(self.action_list[action], self.frames_to_skip)
            terminal = self.env.is_episode_finished()
            state = self.env.get_state()
            new_frame = state.screen_buffer if state is not None else self.frame
            shaping_reward = self.reward_shaper.calc_reward(state.game_variables) \
                if self.reward_shaper is not None and state is not None else 0.0
        else:
            self.env.set_action(self.action_list[action])
            reward = 0.0
            new_frame = self.frame
            terminal = self.env.is_episode_finished()
            new_vars = self.env.get_state().game_variables
            for _ in range(self.frames_to_skip):
                self.env.advance_action()
                terminal = self.env.is_episode_finished()
                if terminal:
                    break
                else:
                    reward += self.env.get_last_reward()
                    state = self.env.get_state()
                    new_frame = state.screen_buffer
                    new_vars = state.game_variables
            shaping_reward = self.reward_shaper.calc_reward(new_vars) \
                if self.reward_shaper is not None else 0.0

        processed_frame = process_frame(new_frame, self.preprocess_shape)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=-1)

        return processed_frame, reward, terminal, shaping_reward

    def stop(self):
        """
        Stop the VizDoom gracefully.
        :return: None
        """
        self.env.close()


def test_game_wrapper():
    g = GameWrapper(
        visible=True, is_sync=True,
        scenario_cfg_path="../scenarios/basic.cfg",
        action_list=[
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [False, False, True],
        ],
        preprocess_shape=(30, 40),
    )
    g.reset()
    print("state shape: ", g.state.shape)

    for e in range(5):
        g.reset()
        for i in range(100):
            s, r, t = g.step(np.random.randint(0, len(g.action_list)))
            print(s.shape, r, t)
            if t:
                break
        print("End of episode {}".format(e))


if __name__ == '__main__':
    test_game_wrapper()
