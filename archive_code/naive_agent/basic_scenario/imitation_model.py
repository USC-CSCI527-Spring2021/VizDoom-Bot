#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : imitation_model.py
# @Author: harry
# @Date  : 1/27/21 10:57 PM
# @Desc  : A naive model for imitation learning based on CNN

import tensorflow as tf

from tensorflow.keras import datasets, layers, models


def build_imitation_model(height: int, width: int, num_channels: int, num_actions: int):
    model = models.Sequential()

    # CNN
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(height, width, num_channels)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(16, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    # FC
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_actions))

    return model


def test_imitation_model():
    model = build_imitation_model(480, 640, 1, 3)
    model.summary()


if __name__ == '__main__':
    test_imitation_model()
