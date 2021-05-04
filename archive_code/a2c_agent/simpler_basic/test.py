#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py.py
# @Author: harry
# @Date  : 2/12/21 1:36 AM
# @Desc  : Description goes here

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import numpy as np
import tensorflow as tf

from a2c_common.model import ActorCritic
from a2c_common.loss import compute_loss


def test():
    model = ActorCritic(num_actions=6)
    model.build(input_shape=(None, 120, 120, 4))
    model.summary()

    loss = compute_loss(
        action_dists=np.array([[0.2, 0.8], [0.5, 0.5]]),
        action_probs=np.array([0.8, 0.5]),
        values=np.array([1.0, 2.0]),
        returns=np.array([0.8, 2.2]),
        entropy_coff=0.0001,
    )
    print(loss)


if __name__ == '__main__':
    test()
