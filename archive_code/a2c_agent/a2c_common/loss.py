#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : loss.py.py
# @Author: harry
# @Date  : 2/12/21 2:13 AM
# @Desc  : Compute actor-critic loss with entropy

import numpy as np
import tensorflow as tf

from tensorflow.keras.losses import Huber

_huber_loss = Huber()
_eps = np.finfo(np.float32).eps.item()


def compute_loss(
        action_dists: tf.Tensor,
        action_probs: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor,
        advantages: tf.Tensor,
        entropy_coff: float = 0.001,
        critic_coff: float = 0.5,
):
    """
    Compute actor-critic loss with entropy.
    :param action_dists: shape (batch_size, num_actions), action distributions (raw outputs of policy network).
    :param action_probs: shape (batch_size,), selected action probs of true actions taken by agent for the episode.
    :param values: shape (batch_size,), outputs of value network.
    :param returns: shape (batch_size,), discounted expected rewards for each time step of the episode.
    :param advantages: shape (batch_size,), computed advantages (returns - values);
     provided separately to stop gradients.
    :param entropy_coff: coefficient for entropy loss.
    :param critic_coff: coefficient for critic loss.
    :return: the computed total loss.
    """
    # compute actor loss (policy loss + entropy loss)
    action_log_probs = tf.math.log(tf.clip_by_value(action_probs, _eps, 1.0))
    policy_loss = tf.math.reduce_mean(action_log_probs * advantages)
    action_log_dists = tf.math.log(tf.clip_by_value(action_dists, _eps, 1.0))
    entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_dists * action_log_dists, axis=-1))
    # minus sign comes from the fact that we want to MAXIMIZE the expected discounted rewards
    actor_loss = -(policy_loss + entropy_coff * entropy_loss)
    actor_loss = tf.cast(actor_loss, dtype='float32')
    print(f'policy_loss: {policy_loss}')
    print(f'entropy_loss: {entropy_loss}')
    print(f'actor_loss: {actor_loss}')

    # compute critic loss
    critic_loss = tf.cast(_huber_loss(returns, values), dtype='float32')
    print(f'critic_loss: {critic_loss}')

    return actor_loss + critic_coff * critic_loss


def compute_loss_ppo(
        action_dists: tf.Tensor,
        action_probs: tf.Tensor,
        action_probs_old: tf.Tensor,
        values: tf.Tensor,
        returns: tf.Tensor,
        advantages: tf.Tensor,
        entropy_coff: float = 0.001,
        critic_coff: float = 0.5,
        epsilon: float = 0.2,
):
    """
    Compute actor-critic loss with entropy using PPO.
    Ref: https://towardsdatascience.com/proximal-policy-optimization-ppo-with-tensorflow-2-x-89c9430ecc26
    :param action_dists: shape (batch_size, num_actions), action distributions (raw outputs of policy network).
    :param action_probs: shape (batch_size,), selected action probs of true actions
     taken by agent (with policy updated by optimizer) for the episode.
    :param action_probs_old: shape (batch_size,), selected action probs of true actions
     taken by agent (with policy before updated) for the episode.
    :param values: shape (batch_size,), outputs of value network.
    :param returns: shape (batch_size,), discounted expected rewards for each time step of the episode.
    :param advantages: shape (batch_size,), computed advantages (returns - values);
     provided separately to stop gradients.
    :param entropy_coff: coefficient for entropy loss.
    :param critic_coff: coefficient for critic loss.
    :param epsilon: PPO policy distribution ratio clipping hyperparameter.
    :return: the computed total loss.
    """
    # compute actor loss (policy loss + entropy loss)
    ratios = tf.clip_by_value(action_probs, _eps, 1.0) / tf.clip_by_value(action_probs_old, _eps, 1.0)
    policy_loss_no_clip = ratios * advantages
    policy_loss_clip = tf.clip_by_value(ratios, 1.0 - epsilon, 1.0 + epsilon) * advantages
    policy_loss = tf.reduce_mean(tf.math.minimum(policy_loss_no_clip, policy_loss_clip))
    action_log_dists = tf.math.log(tf.clip_by_value(action_dists, _eps, 1.0))
    entropy_loss = -tf.reduce_mean(tf.reduce_sum(action_dists * action_log_dists, axis=-1))
    # minus sign comes from the fact that we want to MAXIMIZE the expected discounted rewards
    actor_loss = -(policy_loss + entropy_coff * entropy_loss)
    actor_loss = tf.cast(actor_loss, dtype='float32')
    # print(f'ratios: {ratios}')
    # print(f'policy_loss_no_clip: {policy_loss_no_clip}')
    # print(f'policy_loss_clip: {policy_loss_clip}')
    print(f'policy_loss: {policy_loss}')
    print(f'entropy_loss: {entropy_loss}')
    print(f'actor_loss: {actor_loss}')

    # compute critic loss
    critic_loss = tf.cast(_huber_loss(returns, values), dtype='float32')
    print(f'critic_loss: {critic_loss}')

    return actor_loss + critic_coff * critic_loss
