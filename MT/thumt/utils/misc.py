# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

_GLOBAL_STEP = 0
_GLOBAL_TIME = 0.


def get_global_step():
    return _GLOBAL_STEP

def set_global_step(step):
    global _GLOBAL_STEP
    _GLOBAL_STEP = step

def get_global_time():
    return _GLOBAL_TIME

def set_global_time(time):
    global _GLOBAL_TIME
    _GLOBAL_TIME = time
