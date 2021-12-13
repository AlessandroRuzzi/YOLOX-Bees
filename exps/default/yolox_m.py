#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.num_classes = 1
        self.data_dir = "datasets/bees_all/"
        self.train_ann = "train.json"
        self.val_ann = "val.json"
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
