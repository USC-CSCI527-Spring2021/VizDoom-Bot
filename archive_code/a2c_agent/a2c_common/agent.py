#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : agent.py.py
# @Author: harry
# @Date  : 2/12/21 3:27 AM
# @Desc  : A2C Agent

import os
import numpy as np
import tensorflow as tf

from a2c_common.model import ActorCritic
from a2c_common.game_wrapper import GameWrapper
from a2c_common.utils import get_expected_return, generalized_advantage_estimation
from a2c_common.loss import compute_loss, compute_loss_ppo
from typing import Tuple, List


class A2CAgent(object):
    def __init__(
            self,
            model: ActorCritic,
            game: GameWrapper,
            num_actions: int,
            # input_shape: Tuple[int, int] = (120, 120),
    ):
        self.model = model
        self.game = game
        self.num_actions = num_actions
        # self.input_shape = input_shape

    def set_game_wrapper(self, game: GameWrapper):
        self.game = game

    def get_action(self, state: np.ndarray, stochastic: bool = True) -> Tuple[int, np.ndarray]:
        """
        Get action to take given current state
        :param state: game state of shape (height, width, num_channels).
        :param stochastic: randomly sample an action from policy probs if True, choose argmax action o.w.
        :return: chosen action id, along with distribution of actions (output of policy network)
        """
        state = tf.expand_dims(state, 0)
        action_probs, _ = self.model(state)
        # print(f'action_probs: {action_probs}')

        if stochastic:
            action = tf.random.categorical(action_probs, 1)[0, 0]
        else:
            action = tf.math.argmax(action_probs, -1)[0]

        return int(action), action_probs

    def run_batch(self, batch_size: int, reward_shaping: bool = False) -> List[tf.Tensor]:
        """
        Runs a single batch to collect training data of length batch_size.
        :param batch_size
        :param reward_shaping
        :return:
        """
        states = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        actions = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
        # action_probs_raw = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        # action_probs = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        values = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        rewards = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        dones = tf.TensorArray(dtype=tf.bool, size=0, dynamic_size=True)

        state = self.game.get_state()
        state_shape = state.shape
        for t in tf.range(batch_size):
            states = states.write(t, state)

            # run the model and to get action probabilities and critic value
            state = tf.expand_dims(state, axis=0)
            action_dist_t, value = self.model(state)  # action_dist_t = action distribution at time t
            # print(action_probs_raw_t)
            # print(value)

            # sample next action from the action probability distribution
            action = tf.random.categorical(action_dist_t, 1)[0, 0]
            actions = actions.write(t, tf.cast(action, tf.int32))
            # action_probs_t = action_probs_raw_t[0, action]

            # action_probs_raw = action_probs_raw.write(t, action_probs_raw_t[0])
            # action_probs = action_probs.write(t, action_probs_t)
            values = values.write(t, tf.squeeze(value))

            # perform action in game env
            state, reward, done, shaping_reward = self.game.tf_step(action)
            state.set_shape(state_shape)
            if reward_shaping:
                reward += shaping_reward

            rewards = rewards.write(t, reward)
            dones = dones.write(t, done)

            # reset game env if necessary
            if done:
                state = self.game.reset()

        states = states.stack()
        actions = actions.stack()
        # action_probs_raw = action_probs_raw.stack()
        # action_probs = action_probs.stack()
        values = values.stack()
        rewards = rewards.stack()
        dones = dones.stack()

        # if the episode didn't end on the last step we need to compute the value for the last state
        if dones[-1]:
            next_value = tf.constant(0.0, dtype=tf.float32)
        else:
            state = tf.expand_dims(state, axis=0)
            _, value = self.model(state)
            next_value = tf.stop_gradient(value[0, 0])

        return states, actions, values, rewards, dones, next_value

    # XXX: currently using tf.function will cause ViZDoom running abnormally...
    # @tf.function
    def train_step(
            self,
            max_steps_per_episode: int,
            batch_size: int,
            optimizer: tf.keras.optimizers.Optimizer,
            gamma: float = 0.99,
            entropy_coff: float = 0.0001,
            critic_coff: float = 0.5,
            reward_shaping: bool = False,
            standardize_adv: bool = True,
            clip_norm: float = 5.0,
    ) -> tf.Tensor:
        """
        Run a model training episode for max_steps_per_episode steps.
        :param max_steps_per_episode
        :param batch_size
        :param optimizer
        :param gamma
        :param entropy_coff
        :param critic_coff
        :param reward_shaping
        :param standardize_adv
        :param clip_norm
        :return:
        """
        # divide episode steps into batches (ignoring remainder)
        batch_n = max_steps_per_episode // batch_size
        episode_reward = tf.constant(0.0, dtype=tf.float32)

        for _ in tf.range(batch_n):
            # run the model for one batch to collect training data
            states, actions, values, rewards, dones, next_value = self.run_batch(
                batch_size, reward_shaping
            )

            # one-hot encodings for actions
            action_one_hots = tf.one_hot(actions, depth=self.num_actions)

            # calculate expected returns and advantages using GAE
            returns, advantages = generalized_advantage_estimation(
                rewards, dones,
                tf.concat([values, tf.expand_dims(next_value, axis=-1)], axis=-1),
                gamma=gamma,
                lmbda=0.95,
                standardize_adv=standardize_adv,
            )

            # print(f"rewards: {rewards}")
            # print(f"dones: {dones}")
            # print(f"next_value: {next_value}")
            # print(f"returns: {returns}")
            # print(f"advantages: {advantages}")

            # convert training data to appropriate TF tensor shapes
            # returns = tf.expand_dims(returns, -1)  # shape = (batch_size, 1)

            with tf.GradientTape() as tape:
                # forward pass
                action_dists, values_pred = self.model(states)

                # collect action_probs from action_dists
                action_probs = tf.reduce_sum(action_dists * action_one_hots, axis=-1)

                # calculate loss
                loss = compute_loss(
                    action_dists, action_probs,
                    values_pred, returns, advantages,
                    entropy_coff, critic_coff,
                )
                print(f"loss: {loss}")

            grads = tape.gradient(loss, self.model.trainable_variables)
            grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
            print(f"global_norm: {global_norm}")
            optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            batch_reward = tf.math.reduce_sum(rewards)
            episode_reward += batch_reward

        return episode_reward

    # XXX: currently using tf.function will cause ViZDoom running abnormally...
    # @tf.function
    def train_step_ppo(
            self,
            max_steps_per_episode: int,
            batch_size: int,
            optimizer: tf.keras.optimizers.Optimizer,
            gamma: float = 0.99,
            entropy_coff: float = 0.0001,
            critic_coff: float = 0.5,
            reward_shaping: bool = False,
            standardize_adv: bool = True,
            clip_norm: float = 5.0,
            epochs_per_batch: int = 10,
            epsilon: float = 0.2,
    ) -> tf.Tensor:
        """
        Run a model training episode for max_steps_per_episode steps using PPO loss.
        :param max_steps_per_episode
        :param batch_size
        :param optimizer
        :param gamma
        :param entropy_coff
        :param critic_coff
        :param reward_shaping
        :param standardize_adv
        :param clip_norm
        :param epochs_per_batch: how many epochs of training to run for each batch.
        :param epsilon: used for PPO ratio clipping.
        :return:
        """
        # divide episode steps into batches (ignoring remainder)
        batch_n = max_steps_per_episode // batch_size
        episode_reward = tf.constant(0.0, dtype=tf.float32)

        for _ in tf.range(batch_n):
            # run the model for one batch to collect training data
            states, actions, values, rewards, dones, next_value = self.run_batch(
                batch_size, reward_shaping
            )

            # one-hot encodings for actions
            action_one_hots = tf.one_hot(actions, depth=self.num_actions)

            # get action_probs before updating the network
            action_dists, _ = self.model(states)
            action_probs_old = tf.reduce_sum(action_dists * action_one_hots, axis=-1)

            # calculate expected returns and advantages using GAE
            returns, advantages = generalized_advantage_estimation(
                rewards, dones,
                tf.concat([values, tf.expand_dims(next_value, axis=-1)], axis=-1),
                gamma=gamma,
                lmbda=0.95,
                standardize_adv=standardize_adv,
            )

            # print(f"rewards: {rewards}")
            # print(f"values: {values}")
            # print(f"dones: {dones}")
            # print(f"next_value: {next_value}")
            # print(f"returns: {returns}")
            # print(f"advantages: {advantages}")

            # convert training data to appropriate TF tensor shapes
            # returns = tf.expand_dims(returns, -1)  # shape = (batch_size, 1)

            # train the network on this batch for a number of epochs
            for _ in range(epochs_per_batch):
                with tf.GradientTape() as tape:
                    # forward pass
                    action_dists, values_pred = self.model(states)

                    # collect action_probs from action_dists
                    action_probs = tf.reduce_sum(action_dists * action_one_hots, axis=-1)

                    # calculate loss
                    loss = compute_loss_ppo(
                        action_dists, action_probs, action_probs_old,
                        values_pred, returns, advantages,
                        entropy_coff, critic_coff, epsilon,
                    )
                    print(f"loss: {loss}")

                grads = tape.gradient(loss, self.model.trainable_variables)
                grads, global_norm = tf.clip_by_global_norm(grads, clip_norm)
                print(f"global_norm: {global_norm}")
                optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            batch_reward = tf.math.reduce_sum(rewards)
            episode_reward += batch_reward

        return episode_reward

    def evaluation_step(
            self,
            stochastic: bool = True,
    ) -> float:
        """
        Perform one evaluation episode and return reward.
        :param stochastic: whether use stochastic policy or deterministic policy.
        :return: evaluation reward.
        """
        state = self.game.reset()
        reward = 0.0
        done = False
        while not done:
            action, _ = self.get_action(state, stochastic)
            state, r, done, _ = self.game.step(action, smooth_rendering=True)
            reward += float(r)
        return reward

    def save(self, folder_name):
        """
        Save model.
        """
        if folder_name is None:
            print('WARNING: folder_name not given, skipping save')
            return

        if not os.path.isdir(folder_name):
            os.makedirs(folder_name)

        # self.model.save(os.path.join(folder_name, 'a2c'), save_format='tf')
        self.model.save_weights(os.path.join(folder_name, 'a2c'), save_format='tf')

    def load(self, folder_name):
        """
        Load model.
        """
        if not os.path.isdir(folder_name):
            raise ValueError(f'{folder_name} is not a valid directory')

        # self.model = tf.keras.models.load_model(os.path.join(folder_name, 'a2c'))
        self.model.load_weights(os.path.join(folder_name, 'a2c'))
