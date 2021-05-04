#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : record_evaluate.py.py
# @Author: harry
# @Date  : 2/14/21 7:15 PM
# @Desc  : Evaluate and record result as video

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import io
import tqdm
import cv2
import vizdoom as vzd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from a2c_common.model import ActorCritic
from a2c_common.agent import A2CAgent
from a2c_common.game_wrapper import GameWrapper
from a2c_common.utils import get_img_from_fig
from constants import *
from params import *
from datetime import datetime


def record_evaluate(
        episodes_to_eval: int = 1,
        stochastic: bool = True,
        filename: str = './evaluation.mp4',
):
    # create game env
    game = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        (RESIZED_HEIGHT, RESIZED_WIDTH),
        FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=True,
        is_sync=False,
        reward_shaper=None,
        screen_format=vzd.ScreenFormat.CRCGCB
    )

    # create and build model
    model = ActorCritic(num_actions=NUM_ACTIONS)
    model.build(input_shape=(None, RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    model.summary()

    # create agent and try to load model
    agent = A2CAgent(model, game, NUM_ACTIONS)
    if LOAD_PATH is None or not os.path.exists(LOAD_PATH):
        print('WARNING: No saved model found, evaluating untrained model')
    else:
        agent.load(LOAD_PATH)
        print(f'Model loaded from {LOAD_PATH}')

    # bootstrap the network
    random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    agent.get_action(random_state)

    # begin recording
    print("Evaluating...")
    frames = []
    action_probs = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            state = game.reset()
            done = False
            while not done:
                action, action_prob = agent.get_action(state, stochastic)
                state, r, done, _, new_frames = game.step(
                    action, smooth_rendering=True, return_frames=True)
                frames.extend(new_frames)
                action_probs.extend([action_prob[0]] * len(new_frames))
    game.env.close()

    # generate prob histograms along with frames
    print("Composing frames...")
    rst_frames = []
    # action_names = ["None", "A", "R", "RA", "L", "LA", "LR", "LRA"]
    action_names = ["None", "L", "R", "A"]
    assert len(action_names) == NUM_ACTIONS
    for f, p in tqdm.tqdm(zip(frames, action_probs), total=len(frames)):
        fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]})
        fig.tight_layout(pad=0.2)
        ax[0].imshow(f.transpose(1, 2, 0))  # (ch, h, w) -> (h, w, ch)
        ax[1].bar(action_names, p)
        ax[1].set_ylim(0.0, 1.0)  # use fixed y axis to stabilize animation
        rst_frame = get_img_from_fig(fig, 180, rgb=False)
        # print(rst_frame.shape)
        plt.close(fig)
        rst_frames.append(rst_frame)

    print("Saving video...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    h = rst_frames[0].shape[0]
    w = rst_frames[0].shape[1]
    out = cv2.VideoWriter(filename, fourcc, 30, (w, h))
    for f in rst_frames:
        out.write(f)
    out.release()


if __name__ == '__main__':
    record_evaluate(1, True, f'evaluation_{datetime.now().strftime("%s")}.mp4')
