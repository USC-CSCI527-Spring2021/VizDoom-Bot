#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: harry
# @Date  : 2/4/21 6:47 PM
# @Desc  : DQN model
# Ref: https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a

import tensorflow as tf
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import (Add, Conv2D, Dense, Flatten, Input,
                                     Lambda, Subtract)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, RMSprop
from constants import RESIZED_HEIGHT, RESIZED_WIDTH, NUM_ACTIONS


def build_q_network(
        n_actions, learning_rate=0.00001,
        input_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        history_length=4
):
    """Builds a dueling DQN as a Keras model
    Arguments:
        n_actions: Number of possible action the agent can take
        learning_rate: Learning rate
        input_shape: Shape of the preprocessed frame the model sees
        history_length: Number of historical frames the agent can see
    Returns:
        A compiled Keras model
    """
    model_input = Input(shape=(input_shape[0], input_shape[1], history_length))
    x = Lambda(lambda layer: layer / 255)(model_input)  # normalize by 255

    x = Conv2D(8, (6, 6), strides=4, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(16, (4, 4), strides=2, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(16, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)
    x = Conv2D(32, (3, 3), strides=1, kernel_initializer=VarianceScaling(scale=2.),
               activation='relu', use_bias=False)(x)

    # Split into value and advantage streams
    val_stream, adv_stream = Lambda(lambda w: tf.split(w, 2, 3))(x)  # custom splitting layer

    val_stream = Flatten()(val_stream)
    val = Dense(1, kernel_initializer=VarianceScaling(scale=2.))(val_stream)

    adv_stream = Flatten()(adv_stream)
    adv = Dense(n_actions, kernel_initializer=VarianceScaling(scale=2.))(adv_stream)

    # Combine streams into Q-Values
    reduce_mean = Lambda(lambda w: tf.reduce_mean(w, axis=1, keepdims=True))  # custom layer for reduce mean
    q_vals = Add()([val, Subtract()([adv, reduce_mean(adv)])])

    # Build model
    model = Model(model_input, q_vals)
    model.compile(Adam(learning_rate), loss=tf.keras.losses.Huber())

    return model


def test_model():
    m = build_q_network(NUM_ACTIONS)
    m.summary()


if __name__ == '__main__':
    test_model()
