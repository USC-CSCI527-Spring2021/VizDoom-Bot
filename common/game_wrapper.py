#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper.py
# @Author: harry
# @Date  : 2/4/21 7:15 PM
# @Desc  : A wrapper class for VizDoom game

import gym
import vizdoom as vzd
import numpy as np

from common.utils import process_frame
from common.i_reward_shaper import IRewardShaper
from typing import List, Tuple
from gym import spaces


class DoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            scenario_cfg_path: str,
            action_list: List[List[bool]],
            preprocess_shape=(120, 120),
            frames_to_skip=4,
            history_length=4,
            visible=False,
            is_sync=True,
            reward_shaper: IRewardShaper = None,
            screen_format=None,
            use_attention=False,
            attention_ratio=0.5,
    ):
        super(DoomEnv, self).__init__()
        # vizdoom game init
        game = vzd.DoomGame()
        game.load_config(scenario_cfg_path)
        game.set_window_visible(visible)
        if is_sync:
            game.set_mode(vzd.Mode.PLAYER)
        else:
            game.set_mode(vzd.Mode.ASYNC_PLAYER)
        if screen_format is None:
            game.set_screen_format(vzd.ScreenFormat.GRAY8)
        else:
            game.set_screen_format(screen_format)
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
        self.use_attention = use_attention
        self.attention_ratio = attention_ratio
        self.height, self.width = preprocess_shape
        self.num_channels = history_length if not use_attention else history_length * 2

        self.frame = None
        self.state = None
        self.state_attention = None

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(len(action_list))
        # Example for using image as input:
        self.observation_space = spaces.Box(
            low=0, high=255,
            shape=(self.height, self.width, self.num_channels),
            dtype=np.uint8,
        )

    def get_state(self) -> np.ndarray:
        if self.use_attention:
            # stack normal state together with attention state
            return np.concatenate((self.state, self.state_attention), axis=-1)
        else:
            return self.state.copy()

    def reset(self) -> np.ndarray:
        """
        Resets the environment and return initial state.
        """
        self.env.new_episode()
        init_state = self.env.get_state()
        self.frame = init_state.screen_buffer
        if self.reward_shaper is not None:
            self.reward_shaper.reset(init_state.game_variables)

        # For the initial state, we stack the first frame history_length times
        self.state = np.repeat(process_frame(self.frame, self.preprocess_shape), self.history_length, axis=-1)
        if self.use_attention:
            self.state_attention = np.repeat(
                process_frame(self.frame, self.preprocess_shape, zoom_in=True, zoom_in_ratio=self.attention_ratio),
                self.history_length, axis=-1
            )

        return self.get_state()

    def step(self, action: int, smooth_rendering=False) \
            -> Tuple[np.ndarray, float, bool, dict]:
        """Performs an action and observes the result
        Arguments:
            action: An integer describe action the agent chose
            smooth_rendering: Whether render intermediate states to make game looks smoother;
                skip rendering tics could potentially expedite training
        Returns:
            state: New game state as a result of that action
            reward: The reward for taking that action
            done: Whether the game has ended
            info: Auxiliary info (shaping_reward, raw_frames)
        """
        frames = []
        if not smooth_rendering:
            # make_action will not update(render) skipped tics
            reward = self.env.make_action(self.action_list[action], self.frames_to_skip)
            done = self.env.is_episode_finished()
            state = self.env.get_state()
            new_frame = state.screen_buffer if state is not None else self.frame
            shaping_reward = self.reward_shaper.calc_reward(state.game_variables) \
                if self.reward_shaper is not None and state is not None else 0.0
            frames.append(new_frame)
        else:
            self.env.set_action(self.action_list[action])
            reward = 0.0
            new_frame = self.frame
            done = self.env.is_episode_finished()
            new_vars = self.env.get_state().game_variables
            frames.append(new_frame)
            for _ in range(self.frames_to_skip):
                self.env.advance_action()
                reward += self.env.get_last_reward()
                done = self.env.is_episode_finished()
                if done:
                    break
                else:
                    state = self.env.get_state()
                    new_frame = state.screen_buffer
                    new_vars = state.game_variables
                    frames.append(new_frame)
            shaping_reward = self.reward_shaper.calc_reward(new_vars) \
                if self.reward_shaper is not None else 0.0

        self.frame = new_frame
        processed_frame = process_frame(new_frame, self.preprocess_shape)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=-1)
        if self.use_attention:
            processed_frame_attention = process_frame(
                new_frame, self.preprocess_shape, zoom_in=True, zoom_in_ratio=self.attention_ratio)
            self.state_attention = np.append(self.state_attention[:, :, 1:], processed_frame_attention, axis=-1)

        return self.get_state().astype(np.uint8), \
               reward + shaping_reward, \
               done, \
               {
                   'shaping_reward': shaping_reward,
                   'frames': frames,
               }

    def render(self, mode='human'):
        if mode == 'human':
            pass
        elif mode == 'rgb_array':
            return self.frame.astype(np.uint8)

    def close(self):
        self.env.close()

    def seed(self, seed=None):
        self.env.set_seed(seed)
        self.env.init()


def test_game_wrapper():
    g = DoomEnv(
        visible=True, is_sync=False,
        scenario_cfg_path="../scenarios/simpler_basic.cfg",
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
            s, r, t, info = g.step(np.random.randint(0, len(g.action_list)), smooth_rendering=True)
            print(s.shape, r, t, info.keys())
            if t:
                break
        print("End of episode {}".format(e))


if __name__ == '__main__':
    test_game_wrapper()
