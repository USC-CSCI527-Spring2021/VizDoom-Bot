#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : game_wrapper.py
# @Author: harry
# @Date  : 2/4/21 7:15 PM
# @Desc  : A wrapper class for VizDoom game

import gym
import vizdoom as vzd
import numpy as np
import random

from common.utils import process_frame
from common.i_reward_shaper import IRewardShaper
from typing import List, Tuple, Type, Optional, Union
from gym import spaces


class DoomEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
            self,
            scenario_cfg_path: Union[str, List[str]],
            action_list: List[List[bool]],
            preprocess_shape=(120, 120),
            frames_to_skip=4,
            history_length=4,
            visible=False,
            is_sync=True,
            is_spec=False,
            reward_shaper: Type[IRewardShaper] = None,
            screen_format=None,
            use_attention=False,
            attention_ratio=0.5,
            game_args: str = '',
            game_map: str = '',
            num_bots: int = 0,
            overwrite_episode_timeout: Optional[int] = None,
            extra_features: Optional[List[int]] = None,
            extra_features_norm_factor: Optional[List[Union[int, float]]] = None,
            complete_before_timeout_reward: float = 0.0,
    ):
        super(DoomEnv, self).__init__()
        self.scenario_cfg_path = scenario_cfg_path

        # vizdoom game init
        def _init_game(cfg_path: str):
            game = vzd.DoomGame()
            game.load_config(cfg_path)
            if overwrite_episode_timeout is not None:
                game.set_episode_timeout(overwrite_episode_timeout)
            game.set_window_visible(visible)
            if is_spec:
                game.set_mode(vzd.Mode.ASYNC_SPECTATOR)
            else:
                if is_sync:
                    game.set_mode(vzd.Mode.PLAYER)
                else:
                    game.set_mode(vzd.Mode.ASYNC_PLAYER)
            if screen_format is None:
                game.set_screen_format(vzd.ScreenFormat.GRAY8)
            else:
                game.set_screen_format(screen_format)
            game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
            rs = None
            if reward_shaper is not None:
                rs = reward_shaper()
                game.set_available_game_variables(rs.get_subscribed_game_var_list())
            if len(game_args) > 0:
                game.add_game_args(game_args)
            if len(game_map) > 0:
                game.set_doom_map(game_map)
            game.init()
            return game, rs

        self._init_game_f = _init_game
        self.reward_shaper = None
        if type(self.scenario_cfg_path) is list:
            self.env, self.reward_shaper = self._init_game_f(random.choice(self.scenario_cfg_path))
        else:
            self.env, self.reward_shaper = self._init_game_f(self.scenario_cfg_path)

        self.action_list = action_list
        self.preprocess_shape = preprocess_shape
        self.frames_to_skip = frames_to_skip
        self.history_length = history_length
        self.use_attention = use_attention
        self.attention_ratio = attention_ratio
        self.height, self.width = preprocess_shape
        self.num_channels = history_length if not use_attention else history_length * 2
        # append extra features as last channel
        self.use_extra_feature = False
        self.extra_features = extra_features
        self.extra_features_norm_factor = extra_features_norm_factor
        if extra_features is not None and len(extra_features) > 0:
            assert extra_features_norm_factor is not None and len(extra_features) == len(extra_features_norm_factor), \
                'length of extra_features and extra_features_norm_factor mismatch'
            assert len(extra_features) <= self.height * self.width, 'too many extra features'
            self.num_channels += 1
            self.use_extra_feature = True
        self.num_bots = num_bots

        self.frame = None
        self.state = None
        self.state_attention = None
        self.state_extra_feature = None
        self.complete_before_timeout_reward = complete_before_timeout_reward
        self.frags = 0

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Discrete(len(action_list))
        if self.use_extra_feature:
            self.observation_space = spaces.Box(
                low=0.0, high=1.0,
                shape=(self.height, self.width, self.num_channels),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, self.num_channels),
                dtype=np.uint8,
            )

    def get_state(self) -> np.ndarray:
        if self.use_attention:
            # stack normal state together with attention state
            s = np.concatenate((self.state, self.state_attention), axis=-1)
        else:
            s = self.state.copy()
        if self.use_extra_feature:
            s = np.concatenate((s, self.state_extra_feature), axis=-1)
        return s

    def update_extra_feature(self):
        # retrieve extra features from game variables
        if self.use_extra_feature:
            self.state_extra_feature = np.zeros(self.height * self.width, dtype=np.float32)
            for i, (f, fn) in enumerate(zip(self.extra_features, self.extra_features_norm_factor)):
                self.state_extra_feature[i] = float(self.env.get_game_variable(f)) / float(fn)
            self.state_extra_feature = self.state_extra_feature.reshape((self.height, self.width, 1))

    def reset(self, new_episode: bool = True) -> np.ndarray:
        """
        Resets the environment and return initial state.
        """
        # we need to reinitialize the VizDoom instance if we want to change scenario
        if type(self.scenario_cfg_path) is list:
            self.env, self.reward_shaper = self._init_game_f(random.choice(self.scenario_cfg_path))

        if self.num_bots > 0:
            # Add specific number of bots
            # (file examples/bots.cfg must be placed in the same directory as the Doom executable file,
            # edit this file to adjust bots).
            self.env.send_game_command('removebots')
            for i in range(self.num_bots):
                self.env.send_game_command('addbot')
        if new_episode:
            self.env.new_episode()
        init_state = self.env.get_state()
        self.frame = init_state.screen_buffer
        if self.reward_shaper is not None:
            self.reward_shaper.reset(init_state.game_variables)

        # For the initial state, we stack the first frame history_length times
        self.state = np.repeat(
            process_frame(self.frame, self.preprocess_shape, normalize=self.use_extra_feature),
            self.history_length, axis=-1
        )
        if self.use_attention:
            self.state_attention = np.repeat(
                process_frame(self.frame, self.preprocess_shape,
                              zoom_in=True, zoom_in_ratio=self.attention_ratio,
                              normalize=self.use_extra_feature,
                              ),
                self.history_length, axis=-1
            )
        self.update_extra_feature()

        print(f'FRAGS: {self.frags}')
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
            self.frags = int(self.env.get_game_variable(vzd.GameVariable.FRAGCOUNT))
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
                    self.frags = int(self.env.get_game_variable(vzd.GameVariable.FRAGCOUNT))
            shaping_reward = self.reward_shaper.calc_reward(new_vars) \
                if self.reward_shaper is not None else 0.0

        complete_before_timeout_reward = 0.0
        is_dead = self.env.get_game_variable(vzd.GameVariable.DEAD) > 0.0
        if done and not is_dead and self.env.get_episode_time() <= self.env.get_episode_timeout():
            complete_before_timeout_reward = self.complete_before_timeout_reward

        self.frame = new_frame
        processed_frame = process_frame(new_frame, self.preprocess_shape, normalize=self.use_extra_feature)
        self.state = np.append(self.state[:, :, 1:], processed_frame, axis=-1)
        if self.use_attention:
            processed_frame_attention = process_frame(
                new_frame, self.preprocess_shape, zoom_in=True, zoom_in_ratio=self.attention_ratio,
                normalize=self.use_extra_feature,
            )
            self.state_attention = np.append(self.state_attention[:, :, 1:], processed_frame_attention, axis=-1)
        self.update_extra_feature()

        obs_dtype = np.float32 if self.use_extra_feature else np.uint8
        return self.get_state().astype(obs_dtype), \
               reward + shaping_reward + complete_before_timeout_reward, \
               done, \
               {
                   'shaping_reward': shaping_reward,
                   'frames': frames,
                   'frags': self.frags,
                   'is_dead': is_dead,
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
