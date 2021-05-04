#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py.py
# @Author: harry
# @Date  : 2/12/21 1:15 AM
# @Desc  : Actor-Critic Model with shared weights implemented using tf.keras subclassing API

import tensorflow as tf

from tensorflow.keras import layers
from typing import Tuple


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""

    def __init__(
            self,
            num_actions: int):
        super().__init__()

        # CNN layers
        self.conv1 = layers.Conv2D(32, (7, 7), strides=2, activation='relu')
        self.conv2 = layers.Conv2D(64, (7, 7), strides=2, activation='relu')
        self.mp1 = layers.MaxPool2D((3, 3), strides=2)
        self.conv3 = layers.Conv2D(128, (3, 3), activation='relu')
        self.mp2 = layers.MaxPool2D((3, 3), strides=2)
        self.conv4 = layers.Conv2D(192, (3, 3), activation='relu')

        # flatten
        self.flatten = layers.Flatten()

        # common hidden FC layer
        self.common = layers.Dense(1024, activation="relu")

        # actor branch FC output layer
        self.actor = layers.Dense(num_actions, activation='softmax')
        # critic branch FC output layer
        self.critic = layers.Dense(1, activation=None)

    def call(self, inputs: tf.Tensor, training=None, mask=None) -> Tuple[tf.Tensor, tf.Tensor]:
        # inputs.shape = (batch_size, height, width, num_channels)
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.mp1(x)
        x = self.conv3(x)
        x = self.mp2(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.common(x)

        return self.actor(x), self.critic(x)
