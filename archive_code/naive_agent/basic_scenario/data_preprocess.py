#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_preprocess.py
# @Author: harry
# @Date  : 1/27/21 7:05 PM
# @Desc  : Data preprocessor of raw play data

import numpy as np
import glob
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from constants import RAW_DATA_PATH
from typing import Any, List, Tuple, Optional


def load_raw_data(path: str) -> List[Tuple['np.array', List[float], float]]:
    """
    Load multiple raw play data from path and merge them together.
    :param path: the path containing multiple raw data pickles.
    :return: merged list of raw data.
    """
    if not os.path.exists(path):
        raise RuntimeError("raw data path not exist")

    history = list()
    h_list = glob.glob(os.path.join(path, '*.pkl'))
    for h in h_list:
        with open(h, 'rb') as f:
            history.extend(pickle.load(f))

    return history


def preprocess_raw_data(history: List[Tuple['np.array', List[float], float]]) \
        -> ('np.array', 'np.array'):
    """
    Filtering, normalizing, and concatenating raw data into np arrays.
    :param history: a list of raw data.
    :return: images, labels.
    """
    imgs = list()
    labels = list()  # 0 - LEFT, 1 - RIGHT, 2 - ATTACK

    for h in history:
        i, l, _ = h

        # determine label
        l = np.array(l)
        sum_l = np.sum(l)
        # skip non-action and all-action samples
        if sum_l == 0 or sum_l == 3.0:
            continue
        l_int = 0
        if sum_l == 2.0:
            # prioritize ATTACK
            if l[2] == 1.0:
                l_int = 2
            else:
                l_int = np.random.randint(0, 2)
        else:
            l_int = np.sum(l * np.arange(float(l.shape[0])))

        # normalize img
        i = i.astype(np.float)
        i /= 255.0

        imgs.append(i)
        labels.append(l_int)

    return np.expand_dims(np.stack(imgs, axis=0), axis=-1), np.array(labels, dtype=np.int)


def test_data_preprocess():
    his = load_raw_data(RAW_DATA_PATH)
    print(len(his))
    # samp_i = np.random.randint(0, len(his))
    # print(his[samp_i][0])
    # print(his[samp_i][1])
    # print(his[samp_i][2])
    # print(his[samp_i][0].shape)
    # im = plt.imshow(his[samp_i][0], cmap='gray')
    # plt.show()

    x_train, y_train = preprocess_raw_data(his)
    assert x_train.shape[0] == y_train.shape[0]
    print(x_train.shape)
    print(y_train.shape)
    samp_i = np.random.randint(0, x_train.shape[0])
    print(y_train[samp_i])
    im = plt.imshow(x_train[samp_i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    test_data_preprocess()
