#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py.py
# @Author: harry
# @Date  : 2/4/21 8:53 PM
# @Desc  : Train a DQN agent

import sys
import os

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

import time

import numpy as np
import tensorflow as tf

from dqn_common.game_wrapper import GameWrapper
from dqn_common.replay_buffer import ReplayBuffer
from dqn_common.agent import DQNAgent
from constants import *
from params import *
from model import build_q_network
from reward_shaper import RewardShaper


def train():
    # Create environment
    game_wrapper = GameWrapper(
        SCENARIO_CFG_PATH, ACTION_LIST,
        INPUT_SHAPE, FRAMES_TO_SKIP, HISTORY_LENGTH,
        visible=VISIBLE_TRAINING, is_sync=True,
        reward_shaper=RewardShaper() if USE_REWARD_SHAPING else None,
    )

    # TensorBoard writer
    writer = tf.summary.create_file_writer(TENSORBOARD_DIR)

    # Build main and target networks
    main_dqn = build_q_network(NUM_ACTIONS, LEARNING_RATE, INPUT_SHAPE, HISTORY_LENGTH)
    target_dqn = build_q_network(NUM_ACTIONS, LEARNING_RATE, INPUT_SHAPE, HISTORY_LENGTH)

    replay_buffer = ReplayBuffer(
        MEM_SIZE, INPUT_SHAPE, HISTORY_LENGTH, use_per=USE_PER
    )
    agent = DQNAgent(
        main_dqn, target_dqn, replay_buffer, NUM_ACTIONS,
        INPUT_SHAPE, BATCH_SIZE, HISTORY_LENGTH,
        eps_annealing_frames=EPS_ANNEALING_FRAMES,
        replay_buffer_start_size=MEM_SIZE / 2,
        max_frames=TOTAL_FRAMES,
        use_per=USE_PER,
    )

    # load saved model
    if LOAD_FROM is None or not os.path.exists(LOAD_FROM):
        print("No saved model found, training from scratch")
        frame_number = 0
        rewards = []
        loss_list = []
    else:
        print('Loading from', LOAD_FROM)
        meta = agent.load(LOAD_FROM, LOAD_REPLAY_BUFFER)

        if RESET_META_INFO:
            print("Resetting meta info")
            frame_number = 0
            rewards = []
            loss_list = []
        else:
            # Apply information loaded from meta
            frame_number = meta['frame_number']
            rewards = meta['rewards']
            loss_list = meta['loss_list']

        print('Loaded')

    # Main loop
    try:
        with writer.as_default():
            while frame_number < TOTAL_FRAMES:
                epoch_frame = 0
                while epoch_frame < FRAMES_BETWEEN_EVAL:
                    # begin of episode
                    start_time = time.time()
                    game_wrapper.reset()
                    episode_reward_sum = 0
                    terminal = False
                    while not terminal:
                        # Get action
                        action = agent.get_action(frame_number, game_wrapper.state)

                        # Take step
                        processed_frame, reward, terminal, shaping_reward = \
                            game_wrapper.step(action, smooth_rendering=False)
                        reward += shaping_reward
                        frame_number += 1
                        epoch_frame += 1
                        episode_reward_sum += reward

                        # Add experience to replay memory
                        agent.add_experience(
                            action=action,
                            frame=processed_frame[:, :, 0],
                            reward=reward,
                            terminal=terminal
                        )

                        # perform learning step
                        if frame_number % UPDATE_FREQ == 0 and agent.replay_buffer.count > MIN_REPLAY_BUFFER_SIZE:
                            loss, _ = agent.learn(
                                BATCH_SIZE, gamma=DISCOUNT_FACTOR, frame_number=frame_number,
                                priority_scale=PRIORITY_SCALE
                            )
                            loss_list.append(loss)

                        # Update target network
                        if frame_number % TARGET_UPDATE_FREQ == 0 and frame_number > MIN_REPLAY_BUFFER_SIZE:
                            agent.update_target_network()
                    # end of episode

                    print(".", end='', flush=True)
                    rewards.append(episode_reward_sum)

                    # Output the progress every 10 games
                    if len(rewards) % 10 == 0:
                        # Write to TensorBoard
                        if WRITE_TENSORBOARD:
                            tf.summary.scalar('Reward', np.mean(rewards[-10:]), frame_number)
                            tf.summary.scalar('Loss', np.mean(loss_list[-100:]), frame_number)
                            writer.flush()

                        print("")
                        print(
                            f'Game number: {str(len(rewards)).zfill(6)}  '
                            f'Frame number: {str(frame_number).zfill(8)}  '
                            f'Average reward: {np.mean(rewards[-10:]):0.1f}  '
                            f'Std: {np.std(rewards[-10:]):0.1f}  '
                            f'Min: {np.min(rewards[-10:]):0.1f}  '
                            f'Max: {np.max(rewards[-10:]):0.1f}  '
                            f'Time taken: {(time.time() - start_time):.1f}s  '
                            f'Epsilon: {agent.calc_epsilon(frame_number):.4f}'
                        )

                # Evaluation every `FRAMES_BETWEEN_EVAL` frames
                print("")
                print("Evaluating")
                terminal = True
                eval_rewards = []
                evaluate_frame_number = 0
                for _ in range(EVAL_LENGTH):
                    if terminal:
                        print(".", end='', flush=True)
                        game_wrapper.reset()
                        episode_reward_sum = 0
                        terminal = False

                    action = agent.get_action(frame_number, game_wrapper.state, evaluation=True)
                    _, reward, terminal, _ = game_wrapper.step(action, smooth_rendering=False)
                    evaluate_frame_number += 1
                    episode_reward_sum += reward

                    # On game-over
                    if terminal:
                        eval_rewards.append(episode_reward_sum)

                if len(eval_rewards) > 0:
                    final_score = np.mean(eval_rewards)
                else:
                    # In case the game is longer than the number of frames allowed
                    final_score = episode_reward_sum

                # Print score and write to tensorboard
                print("")
                print('Evaluation score:', final_score)
                if WRITE_TENSORBOARD:
                    tf.summary.scalar('Evaluation score', final_score, frame_number)
                    writer.flush()

                # Save model
                if len(rewards) > 300 and SAVE_PATH is not None:
                    agent.save(
                        SAVE_PATH,
                        frame_number=frame_number, rewards=rewards, loss_list=loss_list
                    )
    except:
        print('\nTraining exited early.')
        writer.close()

        if SAVE_PATH is not None:
            print('Saving...')
            agent.save(
                SAVE_PATH, save_replay_buffer=SAVE_REPLAY_BUFFER,
                frame_number=frame_number, rewards=rewards, loss_list=loss_list
            )
            print('Saved.')

        game_wrapper.stop()


if __name__ == '__main__':
    train()
