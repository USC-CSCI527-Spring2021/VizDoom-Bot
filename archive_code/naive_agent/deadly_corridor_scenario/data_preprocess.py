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

from constants import *
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
    labels = list()

    for h in history:
        img, label, _ = h

        # determine label
        l_int = 0
        label = list(np.array(label, dtype=bool))
        try:
            l_int = ACTION_LIST.index(label)
        except ValueError:
            # for now we skip sample whose action is not in ACTION_LIST
            continue

        # skip non-action sample
        # if l_int == 0:
        #     continue

        # normalize img
        img = img.astype(np.float)
        img /= 255.0

        imgs.append(img)
        labels.append(l_int)

    return np.stack(imgs, axis=0), np.array(labels, dtype=np.int)


def test_data_preprocess():
    his = load_raw_data(RAW_DATA_PATH)
    print('num of raw data samples: ', len(his))
    # samp_i = np.random.randint(0, len(his))
    # print(his[samp_i][0])
    # print(his[samp_i][1])
    # print(his[samp_i][2])
    # print(his[samp_i][0].shape)
    # im = plt.imshow(his[samp_i][0], cmap='gray')
    # plt.show()

    x_train, y_train = preprocess_raw_data(his)
    assert x_train.shape[0] == y_train.shape[0]
    print('x_train.shape: ', x_train.shape)
    print('y_train.shape: ', y_train.shape)
    samp_i = np.random.randint(0, x_train.shape[0])
    print('label of the displayed example: ', y_train[samp_i])
    im = plt.imshow(x_train[samp_i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    test_data_preprocess()
