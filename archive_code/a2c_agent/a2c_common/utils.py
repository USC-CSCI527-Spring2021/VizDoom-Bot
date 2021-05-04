#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2/4/21 6:43 PM
# @Desc  : utils

import cv2
import io
import numpy as np
import tensorflow as tf
import math

from typing import Tuple

# Small epsilon value for stabilizing division operations
_eps = np.finfo(np.float32).eps.item()


def process_frame(frame, shape=(120, 120), normalize=True, zoom_in=False, zoom_in_ratio=0.5):
    """Preprocesses a frame to shape[0] x shape[1] x 1 grayscale
    :param frame: The frame to process.  Must have values ranging from 0-255.
    :param shape: Desired shape to return.
    :param normalize: Whether to normalize the frame by dividing 255.
    :param zoom_in: If true, perform zoom in on the frame and return the zoomed frame.
    :param zoom_in_ratio: Only applicable if zoom_in = True.
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    if len(frame.shape) < 3:
        frame = np.expand_dims(frame, axis=-1)
    else:
        # (ch, h, w) -> (h, w, ch)
        frame = frame.transpose(1, 2, 0)

    if frame.shape[-1] > 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, axis=-1)

    if zoom_in:
        # zoom into the center by cropping
        h, w = frame.shape[0], frame.shape[1]
        dh, dw = math.floor(h * zoom_in_ratio / 2.0), math.floor(w * zoom_in_ratio / 2.0)
        frame = frame[dh:h - dh, dw:w - dw, :]

    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    if normalize:
        frame = frame.astype('float32') / 255.0

    return frame.astype('float32')


def get_img_from_fig(fig, dpi=180, rgb=True):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_expected_return(
        rewards: tf.Tensor,
        dones: tf.Tensor,
        next_value: tf.Tensor,
        gamma: float,
        standardize: bool = True,
) -> tf.Tensor:
    """
    Compute expected returns per timestep.
    :param rewards:
    :param dones:
    :param next_value: the value of the last next state given by the critic network.
    :param gamma
    :param standardize: whether standardize the resulting sequence of returns or not
     (i.e. to have zero mean and unit standard deviation).
    :return:
    """
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    dones = tf.cast(dones[::-1], dtype=tf.float32)

    # TODO: decide whether to use next_value or not
    prev_returns_i = next_value
    # prev_returns_i = 0.0
    # prev_returns_i_shape = prev_returns_i.shape
    for i in tf.range(n):
        returns_i = rewards[i] + gamma * prev_returns_i * (1.0 - dones[i])
        returns = returns.write(i, returns_i)
        prev_returns_i = returns_i
        # prev_returns_i.set_shape(prev_returns_i_shape)

    returns = returns.stack()[::-1]

    if standardize:
        returns = (returns - tf.math.reduce_mean(returns)) / (tf.math.reduce_std(returns) + _eps)

    return returns


def generalized_advantage_estimation(
        rewards: tf.Tensor,
        dones: tf.Tensor,
        extended_values: tf.Tensor,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        standardize_adv: bool = True,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute expected returns and advantages using Generalized Advantage Estimation.
    :return: expected_returns, advantages
    """
    n = tf.shape(rewards)[0]
    returns = tf.TensorArray(dtype=tf.float32, size=n)
    rewards = tf.cast(rewards[::-1], dtype=tf.float32)
    dones = tf.cast(dones[::-1], dtype=tf.float32)
    masks = 1.0 - dones
    extended_values = tf.cast(extended_values[::-1], dtype=tf.float32)
    g = 0

    for i in tf.range(n):
        delta = rewards[i] + gamma * extended_values[i] * masks[i] - extended_values[i + 1]
        g = delta + gamma * lmbda * masks[i] * g
        returns = returns.write(i, g + extended_values[i + 1])

    returns = returns.stack()[::-1]
    extended_values = extended_values[::-1]

    advantages = returns - extended_values[:-1]
    if standardize_adv:
        advantages = (advantages - tf.math.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + _eps)

    return returns, advantages
