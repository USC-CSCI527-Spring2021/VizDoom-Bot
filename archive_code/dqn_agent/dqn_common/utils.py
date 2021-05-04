#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utils.py
# @Author: harry
# @Date  : 2/4/21 6:43 PM
# @Desc  : utils

import cv2
import numpy as np


def process_frame(frame, shape=(60, 80)):
    """Preprocesses a frame to shape[0] x shape[1] x 1 grayscale
    :param frame: The frame to process.  Must have values ranging from 0-255
    :param shape: Desired shape to return
    """
    frame = frame.astype(np.uint8)  # cv2 requires np.uint8, other dtypes will not work

    if len(frame.shape) < 3:
        frame = np.expand_dims(frame, axis=-1)

    if frame.shape[-1] > 1:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    frame = cv2.resize(frame, shape, interpolation=cv2.INTER_NEAREST)
    frame = frame.reshape((*shape, 1))

    return frame
