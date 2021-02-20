#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : i_reward_shaper.py
# @Author: harry
# @Date  : 2/7/21 4:54 PM
# @Desc  : Interface for reward shaping helper class

import numpy as np

from typing import List


class IRewardShaper(object):
    """
    Reward Shaper are used to perform reward shaping during training.
    It should retrieve related game variables from running vizdoom instance
    and calculate shaping reward accordingly.
    """

    def __init__(self):
        pass

    def get_subscribed_game_var_list(self) -> List[int]:
        """
        Return a list of game variables that's essential for this
        reward shaper to perform calculation, so that later they
        are accessible in game states.
        :return: a list of int specifying game variable names.
        """
        raise NotImplemented

    def reset(self, game_vars: 'np.array'):
        """
        Reset internal states given initial game variables.
        :param game_vars: initial game variables.
        :return: None
        """
        raise NotImplemented

    def calc_reward(self, new_game_vars: 'np.array') -> float:
        """
        Calculate shaping reward based on the new game variables
        and internal states.
        :param new_game_vars: new game variables.
        :return: calculated shaping reward.
        """
        raise NotImplemented
