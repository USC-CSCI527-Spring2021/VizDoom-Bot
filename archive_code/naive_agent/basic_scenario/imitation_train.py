#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : imitation_train.py
# @Author: harry
# @Date  : 1/27/21 11:15 PM
# @Desc  : Train the imitation model

import tensorflow as tf
import matplotlib.pyplot as plt

from constants import *
from data_preprocess import *
from imitation_model import build_imitation_model


def imitation_train():
    # hyper params
    batch_size = 32

    # load and preprocess data
    raw_data = load_raw_data(RAW_DATA_PATH)
    x_train, y_train = preprocess_raw_data(raw_data)
    assert x_train.shape[0] == y_train.shape[0]
    num_train_example, height, width, num_channel = x_train.shape
    # we resize images here due to resource limits
    resized_height, resized_width = 120, 160
    x_train = tf.image.resize(x_train, [resized_height, resized_width])

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(1000).batch(batch_size)

    # build, compile, and train model
    model = build_imitation_model(resized_height, resized_width, num_channel, 3)
    model.summary()

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # continue previous ckpt if possible
    prev_ckpt = tf.train.latest_checkpoint(CHECKPOINT_DIR)
    if prev_ckpt is not None:
        print("previous checkpoint detected, loading...")
        model.load_weights(prev_ckpt)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=CHECKPOINT_PATH,
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        monitor='accuracy',
        mode='max',
    )
    history = model.fit(
        train_ds,
        epochs=20,
        callbacks=[cp_callback],
    )

    plt.plot(history.history['accuracy'], label='accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    imitation_train()
