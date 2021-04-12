#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : vec_curiosity_wrapper.py
# @Author: harry
# @Date  : 3/15/21 3:22 AM
# @Desc  : https://github.com/NeoExtended/stable-baselines/blob/master/stable_baselines/common/vec_env/vec_curiosity_reward.py

import logging
import sys

import numpy as np
import tensorflow as tf
from stable_baselines.common import tf_util, tf_layers
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.common.input import observation_input
from stable_baselines.common.running_mean_std import RunningMeanStd
from stable_baselines.common.vec_env import VecEnvWrapper
from gym.spaces import Discrete

from common.vec_tf_wrapper import BaseTFWrapper


# The ICM paper uses 42 x 42 as input. We use 120 x 120. But the CNN below is exactly as stated in the paper.
# May need tuning.
# Also inverse_mapping is a mapping of s_t to a feature vector not the actual inverse model.
# x is TensorFlow tensor. Return is TensorFlow tensor.
def inverse_mapping(x, activ=tf.nn.relu, reuse=False, n_hidden=288, **kwargs):
    with tf.variable_scope("inverse_mapping", reuse=reuse):
        layer_1 = activ(
            tf_layers.conv(x, 'c1', n_filters=32, filter_size=3, stride=2, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        layer_2 = activ(
            tf_layers.conv(
                layer_1, 'c2', n_filters=32, filter_size=3, stride=2, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        layer_3 = activ(
            tf_layers.conv(
                layer_2, 'c3', n_filters=32, filter_size=3, stride=2, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        layer_4 = activ(
            tf_layers.conv(
                layer_3, 'c4', n_filters=32, filter_size=3, stride=2, pad='SAME', init_scale=np.sqrt(2), **kwargs))
        layer_4 = tf_layers.conv_to_fc(layer_4)  # Flattening
        return tf_layers.linear(layer_4, 'fc1', n_hidden=n_hidden, init_scale=np.sqrt(2))


class IcmWrapper(BaseTFWrapper):
    """
    Random Network Distillation (RND) curiosity reward.
    https://arxiv.org/abs/1810.12894
    :param env: (gym.Env) Environment to wrap.
    :param network: (str) Network type. Can be a "cnn" or a "mlp".
    :param intrinsic_reward_weight: (float) Weight for the intrinsic reward.
    :param buffer_size: (int) Size of the replay buffer for predictor training.
    :param train_freq: (int) Frequency of predictor training in steps.
    :param opt_steps: (int) Number of optimization epochs.
    :param batch_size: (int) Number of samples to draw from the replay buffer per optimization epoch.
    :param learning_starts: (int) Number of steps to wait before training the predictor for the first time.
    :param filter_end_of_episode: (bool) Weather or not to filter end of episode signals (dones).
    :param filter_reward: (bool) Weather or not to filter extrinsic reward from the environment.
    :param norm_obs: (bool) Weather or not to normalize and clip obs for the target/predictor network. Note that obs returned will be unaffected.
    :param norm_ext_reward: (bool) Weather or not to normalize extrinsic reward.
    :param gamma: (float) Reward discount factor for intrinsic reward normalization.
    :param learning_rate: (float) Learning rate for the Adam optimizer of the predictor network.
    :param beta: (float) A scalar that weighs the inverse model loss against the forward model loss.
    :param n_hidden: (float) Dimension of the mapped observation space.
    """

    def __init__(self, env,
                 network: str = "cnn",
                 intrinsic_reward_weight: float = 1.0,
                 buffer_size: int = 2048, train_freq: int = 2000, opt_steps: int = 4,
                 batch_size: int = 128, learning_starts: int = 100, filter_end_of_episode: bool = True,
                 filter_reward: bool = False, norm_obs: bool = True,
                 norm_ext_reward: bool = True, gamma: float = 0.99, learning_rate: float = 0.0001,
                 training: bool = True, _init_setup_model=True,
                 beta: float = 0.5, n_hidden: int = 256):

        super().__init__(env, _init_setup_model)

        self.network_type = network
        self.buffer = ReplayBuffer(buffer_size)
        self.train_freq = train_freq
        self.gradient_steps = opt_steps
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.intrinsic_reward_weight = intrinsic_reward_weight  # Equiv to eta in ICM.
        self.filter_end_of_episode = filter_end_of_episode
        self.filter_extrinsic_reward = filter_reward
        self.clip_obs = 5
        self.norm_obs = norm_obs
        self.norm_ext_reward = norm_ext_reward
        self.gamma = gamma  # Discount
        self.learning_rate = learning_rate
        self.training = training

        self.epsilon = 1e-8
        self.int_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.ext_rwd_rms = RunningMeanStd(shape=(), epsilon=self.epsilon)
        self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        self.int_ret = np.zeros(self.num_envs)  # discounted return for intrinsic reward
        self.ext_ret = np.zeros(self.num_envs)  # discounted return for extrinsic reward

        self.updates = 0
        self.steps = 0
        self.last_action = None
        self.last_obs = None
        self.last_update = 0

        self.graph = None
        self.sess = None
        self.observation_ph = None
        self.processed_obs = None
        self.params = None
        self.int_reward = None
        self.aux_loss = None
        self.optimizer = None
        self.training_op = None

        # For ICM:
        assert isinstance(self.action_space, Discrete), 'action_space must be Discrete'
        self.num_actions = self.action_space.n
        self.beta = beta  # weighs the inverse model loss against the forward model loss. Between 1 and 0.
        self.n_hidden = n_hidden
        self.inverse_mapping_current = None
        self.inverse_mapping_next = None
        self.inverse_model = None
        self.forward_model = None
        self.concatenated_obs_inverse = None
        self.concatenated_obs_forward = None
        self.observation_ph_current = None
        self.processed_obs_current = None
        self.observation_ph_next = None
        self.processed_obs_next = None
        self.action_ph = None
        self.action_one_hot = None
        self.inv_loss = None
        self.forward_loss = None

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
        self.graph = tf.Graph()

        with self.graph.as_default():
            self.sess = tf_util.make_session(num_cpu=None, graph=self.graph)

            # placeholders for s_t and s_{t+1}
            # observation_ph_current.shape = (n_batch, width, height, n_channel)
            # observation_ph_next.shape = (n_batch, width, height, n_channel)
            self.observation_ph_current, self.processed_obs_current = observation_input(
                self.venv.observation_space, scale=(self.network_type == "cnn"), name='ob_curr')
            self.observation_ph_next, self.processed_obs_next = observation_input(
                self.venv.observation_space, scale=(self.network_type == "cnn"), name='ob_next')

            # placeholders for a_t
            # action_ph.shape = (n_batch,), i.e. a list of action ids
            # action_one_hot.shape = (n_batch, n_actions)
            # Is data type float32 correct? Is shape correct? Does action need to be processed like obs?
            # What does the self.last_action looks like?? Is it even a vector?? If not should we convert it to vector?
            self.action_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name="Action")
            self.action_one_hot = tf.cast(tf.one_hot(self.action_ph, self.num_actions), tf.float32)

            with tf.variable_scope("inverse_model"):
                # Processed_obs are just placeholders for observation states.
                # inverse_mapping_current.shape = (n_batch, n_hidden)
                # inverse_mapping_next.shape = (n_batch, n_hidden)
                self.inverse_mapping_current = inverse_mapping(
                    self.processed_obs_current, tf.nn.leaky_relu, reuse=False, n_hidden=self.n_hidden)
                self.inverse_mapping_next = inverse_mapping(
                    self.processed_obs_next, tf.nn.leaky_relu, reuse=True, n_hidden=self.n_hidden)
                # Concatenated_obs should be a concatenation of inverse_mapping_current and inverse_mapping_next
                # Is axis=1 correct??
                # concatenated_obs_inverse.shape = (n_batch, 2*n_hidden)
                self.concatenated_obs_inverse = tf.concat(
                    [self.inverse_mapping_current, self.inverse_mapping_next], axis=1)

                # print_op1 = tf.print(
                #     'inverse_mapping_current.shape:', self.inverse_mapping_current.shape, output_stream=sys.stdout)
                # print_op2 = tf.print(
                #     'inverse_mapping_next.shape:', self.inverse_mapping_next.shape, output_stream=sys.stdout)
                # print_op3 = tf.print(
                #     'concatenated_obs_inverse.shape:', self.concatenated_obs_inverse.shape, output_stream=sys.stdout)

                # with tf.control_dependencies([print_op1, print_op2, print_op3]):
                with tf.control_dependencies([]):
                    self.inverse_model = tf_layers.mlp(self.concatenated_obs_inverse, [256])
                    self.inverse_model = tf_layers.linear(self.inverse_model, "predict_action", self.num_actions)
                # inverse_model.shape = (n_batch, n_actions)

            with tf.variable_scope("forward_model"):
                self.concatenated_obs_forward = tf.concat(
                    [self.inverse_mapping_current, self.action_one_hot], axis=1)

                # print_op1 = tf.print(
                #     'action_one_hot.shape:', self.action_one_hot.shape, output_stream=sys.stdout)
                # print_op2 = tf.print(
                #     'self.action_ph.shape:', self.action_ph.shape, output_stream=sys.stdout)
                # print_op3 = tf.print(
                #     'self.concatenated_obs_forward.shape:', self.concatenated_obs_forward.shape,
                #     output_stream=sys.stdout)

                # with tf.control_dependencies([print_op1, print_op2, print_op3]):
                with tf.control_dependencies([]):
                    self.forward_model = tf_layers.mlp(self.concatenated_obs_forward, [256])
                    self.forward_model = tf_layers.linear(
                        self.forward_model, "predict_inverse_mapping_next", self.n_hidden)
                # forward_model.shape = (n_batch, n_hidden)

            with tf.name_scope("loss"):
                # Intrinsic reward:
                # int_reward.shape = (n_batch,)
                self.int_reward = tf.reduce_mean(
                    tf.square(self.forward_model - self.inverse_mapping_next), axis=1)

                # print_op1 = tf.print(
                #     'self.inverse_model.shape:', self.inverse_model.shape, output_stream=sys.stdout)
                # print_op2 = tf.print(
                #     'self.forward_model.shape:', self.forward_model.shape, output_stream=sys.stdout)
                # print_op3 = tf.print(
                #     'self.int_reward.shape:', self.int_reward.shape, output_stream=sys.stdout)

                # with tf.control_dependencies([print_op1, print_op2, print_op3]):
                with tf.control_dependencies([]):
                    # Inverse loss:
                    self.inv_loss = tf.reduce_mean(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.inverse_model, labels=self.action_ph)
                    )
                    # Forward loss:
                    self.forward_loss = tf.reduce_mean(
                        0.5 * tf.square(self.forward_model - self.inverse_mapping_next))
                    # Sum the loss together:
                    self.aux_loss = (1 - self.beta) * self.inv_loss + self.beta * self.forward_loss

                # Reduce_mean with axis = 1? What does this mean? Does it imply that we can have a reward vector?
                # Maybe for multi-objective?

            with tf.name_scope("train"):
                self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                self.training_op = self.optimizer.minimize(self.aux_loss)

            self.params = tf.trainable_variables()
            tf.global_variables_initializer().run(session=self.sess)

    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs

    def step_async(self, actions):
        super().step_async(actions)
        self.last_action = actions
        self.steps += self.num_envs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.buffer.extend(self.last_obs, self.last_action, rews, obs, dones)

        if self.filter_extrinsic_reward:
            rews = np.zeros(rews.shape)
        if self.filter_end_of_episode:
            dones = np.zeros(dones.shape)

        if self.training:
            self.obs_rms.update(obs)

        obs_c = self.normalize_obs(self.last_obs)
        obs_n = self.normalize_obs(obs)

        raw_intrinsic_reward = self.sess.run(
            [self.int_reward],
            {self.observation_ph_current: obs_c, self.observation_ph_next: obs_n,
             self.action_ph: self.last_action},
        )

        if self.training:
            self._update_ext_reward_rms(rews)
            self._update_int_reward_rms(raw_intrinsic_reward)
        # Reward normalization. Section 2.4 of the RDN paper.
        intrinsic_reward = np.array(raw_intrinsic_reward) / np.sqrt(self.int_rwd_rms.var + self.epsilon)
        if self.norm_ext_reward:
            extrinsic_reward = np.array(rews) / np.sqrt(self.ext_rwd_rms.var + self.epsilon)
        else:
            extrinsic_reward = rews

        # In the ICM paper, the intrinsic reward is scaled by eta/2.
        # This is equivalent to intrinsic_reward_weight we have here.
        reward = np.squeeze(extrinsic_reward + self.intrinsic_reward_weight * intrinsic_reward)
        # print("intrinsic_reward: {}".format(intrinsic_reward))
        # print("extrinsic_reward: {}".format(extrinsic_reward))
        # print("reward: {}".format(reward))

        if self.training and self.steps > self.learning_starts and self.steps - self.last_update > self.train_freq:
            self.updates += 1
            self.last_update = self.steps
            self.learn()

        return obs, reward, dones, infos

    def close(self):
        VecEnvWrapper.close(self)

    def learn(self):
        total_loss = 0
        for _ in range(self.gradient_steps):
            obs_batch, act_batch, rews_batch, next_obs_batch, done_mask = self.buffer.sample(self.batch_size)
            # print("Action batch shape:")
            # print(act_batch.shape)
            obs_batch = self.normalize_obs(obs_batch)
            next_obs_batch = self.normalize_obs(next_obs_batch)
            train, loss = self.sess.run(
                [self.training_op, self.aux_loss],
                {self.observation_ph_current: obs_batch, self.observation_ph_next: next_obs_batch,
                 self.action_ph: act_batch},
            )
            total_loss += loss
        # logging.info("Trained predictor. Avg loss: {}".format(total_loss / self.gradient_steps))
        print("[IcmWrapper] Trained predictor. Avg loss: {}".format(total_loss / self.gradient_steps))

    def _update_int_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.int_ret = self.gamma * self.int_ret + reward
        self.int_rwd_rms.update(self.int_ret)

    def _update_ext_reward_rms(self, reward: np.ndarray) -> None:
        """Update reward normalization statistics."""
        self.ext_ret = self.gamma * self.ext_ret + reward
        self.ext_rwd_rms.update(self.ext_ret)

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """
        Normalize observations using observations statistics.
        Calling this method does not update statistics.
        """
        if self.norm_obs:
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                          -self.clip_obs,
                          self.clip_obs)
        return obs

    def get_parameter_list(self):
        return self.params

    def save(self, save_path):
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # self.saver.save(self.sess, save_path)

        data = {
            'network': self.network_type,
            'intrinsic_reward_weight': self.intrinsic_reward_weight,
            'buffer_size': self.buffer.buffer_size,
            'train_freq': self.train_freq,
            'gradient_steps': self.gradient_steps,
            'batch_size': self.batch_size,
            'learning_starts': self.learning_starts,
            'filter_end_of_episode': self.filter_end_of_episode,
            'filter_extrinsic_reward': self.filter_extrinsic_reward,
            'norm_obs': self.norm_obs,
            'norm_ext_reward': self.norm_ext_reward,
            'gamma': self.gamma,
            'learning_rate': self.learning_rate,
            'int_rwd_rms': self.int_rwd_rms,
            'ext_rwd_rms': self.ext_rwd_rms,
            'obs_rms': self.obs_rms
        }

        params_to_save = self.get_parameters()
        self._save_to_file_zip(save_path, data=data, params=params_to_save)
