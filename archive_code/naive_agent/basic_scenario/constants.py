#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : constants.py
# @Author: harry
# @Date  : 1/27/21 7:08 PM
# @Desc  : Description goes here

import os

RAW_DATA_PATH = "./spec_raw_data/"
CHECKPOINT_PATH = './model_ckpt/imitation_{epoch:04d}.ckpt'
# CHECKPOINT_PATH = './model_ckpt/imitation.ckpt'
CHECKPOINT_DIR = os.path.dirname(CHECKPOINT_PATH)
