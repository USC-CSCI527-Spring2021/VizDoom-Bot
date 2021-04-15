#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : policies.py
# @Author: harry
# @Date  : 4/14/21 5:44 PM
# @Desc  : Customized policy networks for stable-baselines

import tensorflow as tf

from stable_baselines.common.policies import *


def create_augmented_nature_cnn(n_extra_features):
    """
    ref: https://github.com/hill-a/stable-baselines/issues/133#issuecomment-561805417
    Create and return a function for augmented_nature_cnn
    used in stable-baselines.

    n_extra_features tells how many extra features there
    will be in the image.
    """

    def augmented_nature_cnn(scaled_images, **kwargs):
        """
        Copied from stable_baselines policies.py.
        This is nature CNN head where last channel of the image contains
        direct features.

        :param scaled_images: (TensorFlow Tensor) Image input placeholder
        :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
        :return: (TensorFlow Tensor) The CNN output layer
        """
        activ = tf.nn.relu

        # Take last channel as direct features
        other_features = tf.layers.flatten(scaled_images[..., -1])
        # Take known amount of direct features, rest are padding zeros
        other_features = other_features[:, :n_extra_features]

        scaled_images = scaled_images[..., :-1]

        layer_1 = activ(
            conv(scaled_images, 'cnn1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2), **kwargs))
        layer_2 = activ(conv(layer_1, 'cnn2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
        layer_3 = activ(conv(layer_2, 'cnn3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
        layer_3 = conv_to_fc(layer_3)

        # Append direct features to the final output of extractor
        img_output = activ(linear(layer_3, 'cnn_fc1', n_hidden=512, init_scale=np.sqrt(2)))

        concat = tf.concat((img_output, other_features), axis=1)

        return concat

    return augmented_nature_cnn


class AugmentedLstmPolicy(RecurrentActorCriticPolicy):
    """
    Policy object that implements actor critic, using LSTMs.
    Augmented to use extra features appended as the last channel of observed states.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param net_arch: (list) Specification of the actor-critic policy network architecture. Notation similar to the
        format described in mlp_extractor but with additional support for a 'lstm' entry in the shared network part.
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param n_extra_features: (int) Number of extra features appended as the last channel of observed states
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    recurrent = True

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, layers=None,
                 net_arch=None, act_fun=tf.tanh, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
                 n_extra_features=0,
                 **kwargs):
        # state_shape = [n_lstm * 2] dim because of the cell and hidden states of the LSTM
        super(AugmentedLstmPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch,
            state_shape=(2 * n_lstm,), reuse=reuse,
            scale=(feature_extraction == "cnn"))

        self._kwargs_check(feature_extraction, kwargs)

        if n_extra_features > 0:
            warnings.warn(f"n_extra_features = {n_extra_features}, using augmented cnn extractor.")
            cnn_extractor = create_augmented_nature_cnn(n_extra_features)

        if net_arch is None:  # Legacy mode
            if layers is None:
                layers = [64, 64]
            else:
                warnings.warn("The layers parameter is deprecated. Use the net_arch parameter instead.")

            with tf.variable_scope("model", reuse=reuse):
                if feature_extraction == "cnn":
                    extracted_features = cnn_extractor(self.processed_obs, **kwargs)
                else:
                    extracted_features = tf.layers.flatten(self.processed_obs)
                    for i, layer_size in enumerate(layers):
                        extracted_features = act_fun(linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                                                            init_scale=np.sqrt(2)))
                input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
                masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                             layer_norm=layer_norm)
                rnn_output = seq_to_batch(rnn_output)
                value_fn = linear(rnn_output, 'vf', 1)

                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

            self._value_fn = value_fn
        else:  # Use the new net_arch parameter
            if layers is not None:
                warnings.warn("The new net_arch parameter overrides the deprecated layers parameter.")
            if feature_extraction == "cnn":
                raise NotImplementedError()

            with tf.variable_scope("model", reuse=reuse):
                latent = tf.layers.flatten(self.processed_obs)
                policy_only_layers = []  # Layer sizes of the network that only belongs to the policy network
                value_only_layers = []  # Layer sizes of the network that only belongs to the value network

                # Iterate through the shared layers and build the shared parts of the network
                lstm_layer_constructed = False
                for idx, layer in enumerate(net_arch):
                    if isinstance(layer, int):  # Check that this is a shared layer
                        layer_size = layer
                        latent = act_fun(linear(latent, "shared_fc{}".format(idx), layer_size, init_scale=np.sqrt(2)))
                    elif layer == "lstm":
                        if lstm_layer_constructed:
                            raise ValueError("The net_arch parameter must only contain one occurrence of 'lstm'!")
                        input_sequence = batch_to_seq(latent, self.n_env, n_steps)
                        masks = batch_to_seq(self.dones_ph, self.n_env, n_steps)
                        rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                                     layer_norm=layer_norm)
                        latent = seq_to_batch(rnn_output)
                        lstm_layer_constructed = True
                    else:
                        assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                        if 'pi' in layer:
                            assert isinstance(layer['pi'],
                                              list), "Error: net_arch[-1]['pi'] must contain a list of integers."
                            policy_only_layers = layer['pi']

                        if 'vf' in layer:
                            assert isinstance(layer['vf'],
                                              list), "Error: net_arch[-1]['vf'] must contain a list of integers."
                            value_only_layers = layer['vf']
                        break  # From here on the network splits up in policy and value network

                # Build the non-shared part of the policy-network
                latent_policy = latent
                for idx, pi_layer_size in enumerate(policy_only_layers):
                    if pi_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the policy network.")
                    assert isinstance(pi_layer_size, int), "Error: net_arch[-1]['pi'] must only contain integers."
                    latent_policy = act_fun(
                        linear(latent_policy, "pi_fc{}".format(idx), pi_layer_size, init_scale=np.sqrt(2)))

                # Build the non-shared part of the value-network
                latent_value = latent
                for idx, vf_layer_size in enumerate(value_only_layers):
                    if vf_layer_size == "lstm":
                        raise NotImplementedError("LSTMs are only supported in the shared part of the value function "
                                                  "network.")
                    assert isinstance(vf_layer_size, int), "Error: net_arch[-1]['vf'] must only contain integers."
                    latent_value = act_fun(
                        linear(latent_value, "vf_fc{}".format(idx), vf_layer_size, init_scale=np.sqrt(2)))

                if not lstm_layer_constructed:
                    raise ValueError("The net_arch parameter must contain at least one occurrence of 'lstm'!")

                self._value_fn = linear(latent_value, 'vf', 1)
                # TODO: why not init_scale = 0.001 here like in the feedforward
                self._proba_distribution, self._policy, self.q_value = \
                    self.pdtype.proba_distribution_from_latent(latent_policy, latent_value)
        self._setup_init()

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})
        else:
            return self.sess.run([self.action, self.value_flat, self.snew, self.neglogp],
                                 {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run(self.policy_proba, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs, self.states_ph: state, self.dones_ph: mask})


class AugmentedCnnLstmPolicy(AugmentedLstmPolicy):
    """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction.
    Augmented to use extra features appended as the last channel of observed states.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param n_extra_features: (int) Number of extra features appended as the last channel of observed states
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch,
                 n_lstm=256, reuse=False,
                 n_extra_features=0,
                 **_kwargs):
        super(AugmentedCnnLstmPolicy, self).__init__(
            sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm, reuse,
            layer_norm=False, feature_extraction="cnn",
            n_extra_features=n_extra_features,
            **_kwargs)
