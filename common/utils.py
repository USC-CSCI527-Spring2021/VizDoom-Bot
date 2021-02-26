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

from typing import Tuple, Dict, List, Any, Callable

# Small epsilon value for stabilizing division operations
_eps = np.finfo(np.float32).eps.item()


def process_frame(frame, shape=(120, 120), zoom_in=False, zoom_in_ratio=0.5):
    """Preprocesses a frame to shape[0] x shape[1] x 1 grayscale
    :param frame: The frame to process.  Must have values ranging from 0-255.
    :param shape: Desired shape to return.
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

    # if normalize:
    #     frame = frame.astype(np.float32) / 255.0

    return frame.astype(np.uint8)


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


def linear_schedule(initial_value: float, end_value: float, verbose: bool = False):
    """
    Linear learning rate schedule.
    :param initial_value
    :param end_value
    :return: (function)
    """

    def func(progress):
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress: (float)
        :return: (float)
        """
        lr = max(progress * initial_value, end_value)
        if verbose:
            print(f'learning_rate: {lr}')
        return lr

    return func


def collect_kv(*from_dicts: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    k2dict_idx = {}
    for i, d in enumerate(from_dicts):
        for k in d.keys():
            if k in k2dict_idx:
                raise ValueError(f'ambiguous key {k} encountered')
            else:
                k2dict_idx[k] = i

    rst = {}
    for k in keys:
        if k not in k2dict_idx:
            raise ValueError(f'unknown key {k} given is keys')
        else:
            rst[k] = from_dicts[k2dict_idx[k]][k]

    return rst


def make_expand_action_f(num_atomic_actions: int) -> Callable:
    def _expand_action(*args) -> List[bool]:
        a = [False] * num_atomic_actions
        for action_id in args:
            if action_id < num_atomic_actions:
                a[action_id] = True
        return a

    return _expand_action
