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

from stable_baselines.common.policies import CnnPolicy
from stable_baselines import PPO2, A2C
from common.game_wrapper import DoomEnv
from common.utils import get_img_from_fig
from constants import *
from params import *
from datetime import datetime


def record_evaluate(
        episodes_to_eval: int = 1,
        stochastic: bool = True,
        filename: str = './evaluation.mp4',
):
    eval_env = DoomEnv(
        scenario_cfg_path=SCENARIO_CFG_PATH,
        action_list=ACTION_LIST,
        preprocess_shape=(RESIZED_HEIGHT, RESIZED_WIDTH),
        frames_to_skip=FRAMES_TO_SKIP,
        history_length=HISTORY_LENGTH,
        visible=True,
        is_sync=True,
        screen_format=vzd.ScreenFormat.CRCGCB,
        use_attention=USE_ATTENTION,
        attention_ratio=ATTENTION_RATIO,
    )

    try:
        agent = PPO2.load(LOAD_PATH)
        print("Model loaded")
    except:
        print("Failed to load model, evaluate untrained model...")
        agent = PPO2(
            CnnPolicy, eval_env, verbose=True,
        )

    # bootstrap the network
    if USE_ATTENTION:
        random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH * 2))
    else:
        random_state = np.random.normal(size=(RESIZED_HEIGHT, RESIZED_WIDTH, HISTORY_LENGTH))
    agent.predict(random_state)

    frames = []
    rewards = []
    action_probs = []
    with tqdm.trange(episodes_to_eval) as t:
        for i in t:
            episode_r = 0.0
            state = eval_env.reset()
            done = False
            while not done:
                action, _ = agent.predict(state, deterministic=not stochastic)
                action_prob = agent.action_probability(state)
                state, step_r, done, info = eval_env.step(action, smooth_rendering=True)
                frames.extend(info['frames'])
                action_probs.extend([action_prob] * len(info['frames']))
                episode_r += step_r
            rewards.append(episode_r)
            t.set_description(f'Episode {i}')
            t.set_postfix(
                episode_reward=episode_r)
    rewards = np.array(rewards, dtype=np.float32)
    print(f'avg: {rewards.mean()}, std: {rewards.std()}, min: {rewards.min()}, max: {rewards.max()}')
    eval_env.close()

    # generate prob histograms along with frames
    print("Composing frames...")
    rst_frames = []
    action_names = ["N", "L", "R", "A"]
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
