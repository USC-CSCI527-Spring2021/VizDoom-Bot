#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : replay_buffer.py
# @Author: harry
# @Date  : 2/4/21 8:12 PM
# @Desc  : Prioritized replay buffer for DQN agent

import shutil
import os
import random
import numpy as np


class ReplayBuffer:
    """Replay Buffer to store transitions.
    This implementation was heavily inspired by Fabio M. Graetz's replay buffer
    here: https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb"""

    def __init__(
            self, size=10_000,
            input_shape=(60, 80),
            history_length=4, use_per=True
    ):
        """
        Arguments:
            size: Integer, Number of stored transitions
            input_shape: Shape of the preprocessed frame
            history_length: Integer, Number of frames stacked together to create a state for the agent
            use_per: Use PER instead of classic experience replay
        """
        self.size = size
        self.input_shape = input_shape
        self.history_length = history_length
        self.count = 0  # total index of memory written to, always less than self.size
        self.current = 0  # index to write to

        # Pre-allocate memory
        self.actions = np.empty(self.size, dtype=np.int32)
        self.rewards = np.empty(self.size, dtype=np.float32)
        self.frames = np.empty((self.size, self.input_shape[0], self.input_shape[1]), dtype=np.uint8)
        self.terminal_flags = np.empty(self.size, dtype=np.bool)
        self.priorities = np.zeros(self.size, dtype=np.float32)

        self.use_per = use_per

    def add_experience(self, action, frame, reward, terminal):
        """Saves a transition to the replay buffer
        Arguments:
            action: An integer between 0 and env.action_space.n - 1
                determining the action the agent performed
            frame: A frame of the game in grayscale
            reward: A float determining the reward the agent received for performing an action
            terminal: A bool stating whether the episode terminated
        """
        if frame.shape != self.input_shape:
            raise ValueError('frame.shape mismatches input_shape')

        # Write memory
        self.actions[self.current] = action
        self.frames[self.current, ...] = frame
        self.rewards[self.current] = reward
        self.terminal_flags[self.current] = terminal
        self.priorities[self.current] = max(self.priorities.max(), 1)  # make the most recent experience important
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.size

    def get_minibatch(self, batch_size=32, priority_scale=0.0):
        """Returns a minibatch of self.batch_size = 32 transitions
        Arguments:
            batch_size: How many samples to return
            priority_scale: How much to weight priorities. 0 = completely random, 1 = completely based on priority
        Returns:
            A tuple of states, actions, rewards, new_states, and terminals
            If use_per is True:
                An array describing the importance of transition. Used for scaling gradient steps.
                An array of each index that was sampled
        """
        if self.count < self.history_length:
            raise ValueError('Not enough memories to get a minibatch')

        # Get sampling probabilities from priority list
        sample_probabilities = None
        if self.use_per:
            scaled_priorities = self.priorities[self.history_length:self.count - 1] ** priority_scale
            sample_probabilities = scaled_priorities / sum(scaled_priorities)

        # Get a list of valid indices
        indices = []
        for i in range(batch_size):
            while True:
                # Get a random number from history_length
                # to maximum frame written with probabilities based on priority weights
                if self.use_per:
                    index = np.random.choice(np.arange(self.history_length, self.count - 1), p=sample_probabilities)
                else:
                    index = random.randint(self.history_length, self.count - 1)

                # We check that all frames are from same episode with the two following if statements.
                # If either is True, the index is invalid.
                if index >= self.current >= index - self.history_length:
                    continue
                if self.terminal_flags[index - self.history_length:index].any():
                    continue
                break
            indices.append(index)

        # Retrieve states from memory
        states = []
        new_states = []
        for idx in indices:
            states.append(self.frames[idx - self.history_length:idx, ...])
            new_states.append(self.frames[idx - self.history_length + 1:idx + 1, ...])

        # (batch_size, height, width, n_channels)
        states = np.transpose(np.asarray(states), axes=(0, 2, 3, 1))
        new_states = np.transpose(np.asarray(new_states), axes=(0, 2, 3, 1))

        if self.use_per:
            # Get importance weights from probabilities calculated earlier
            importance = 1 / self.count * 1 / sample_probabilities[[index - self.history_length for index in indices]]
            importance = importance / importance.max()

            return (states, self.actions[indices], self.rewards[indices], new_states,
                    self.terminal_flags[indices]), importance, indices
        else:
            return states, self.actions[indices], self.rewards[indices], new_states, self.terminal_flags[indices]

    def set_priorities(self, indices, errors, offset=0.1):
        """Update priorities for PER
        Arguments:
            indices: Indices to update
            errors: For each index, the error between the target Q-vals and the predicted Q-vals
            offset: small number to avoid 0 priority
        """
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset

    def save(self, folder_name):
        """Save the replay buffer to a folder"""
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)

        for k, obj in zip(
                ['actions', 'frames', 'rewards', 'terminal_flags'],
                [self.actions, self.frames, self.rewards, self.terminal_flags]
        ):
            np.save(os.path.join(folder_name, "{}.npy".format(k)), obj)

    def load(self, folder_name):
        """Loads the replay buffer from a folder"""
        for k, obj in zip(
                ['actions', 'frames', 'rewards', 'terminal_flags'],
                [self.actions, self.frames, self.rewards, self.terminal_flags]
        ):
            obj = np.load(os.path.join(folder_name, "{}.npy".format(k)))
            if k == 'actions':
                self.actions = obj
            elif k == 'frames':
                self.frames = obj
            elif k == 'rewards':
                self.rewards = obj
            elif k == 'terminal_flags':
                self.terminal_flags = obj
        # self.actions = np.load(folder_name + '/actions.npy')
        # self.frames = np.load(folder_name + '/frames.npy')
        # self.rewards = np.load(folder_name + '/rewards.npy')
        # self.terminal_flags = np.load(folder_name + '/terminal_flags.npy')


def test_replay_buffer():
    buf = ReplayBuffer()
    print(buf.frames.shape)
    test_dir = "./test_replay_buffer"
    buf.save(test_dir)
    buf.load(test_dir)
    shutil.rmtree(test_dir)


if __name__ == '__main__':
    test_replay_buffer()
