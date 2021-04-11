# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import logging
import os
import re
import six
import socket
import glob
import json
import pickle
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import thumt.data as data
import thumt.models as models
import thumt.utils as utils


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw mask difference surface.",
        usage="draw_mask_difference.py [<args>] [-h | --help]"
    )

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoints.")
    parser.add_argument("--env", type=str, default="",
                        help="env for visdom")

    return parser.parse_args()

def compute_mask_distance(mask_i, mask_j):
    distance = (mask_i - mask_j).abs().sum().item()
    return distance / mask_i.size(0)

def compute_distance_matrix(masks):
    length = len(masks)
    distance_matrix = torch.zeros(length, length)
    for i in range(length):
        for j in range(length):
            distance_matrix[i, j] = compute_mask_distance(masks[i][1], masks[j][1])
    return distance_matrix

def main(args):

    if not args.env:
        env_name = re.sub("-|/", "", args.checkpoint)
    else:
        env_name = args.env

    import visdom
    vis = visdom.Visdom(env=env_name)
    assert vis.check_connection()

    masks = pickle.load(open(args.checkpoint, "rb"))
    distance_matrix = compute_distance_matrix(masks)

    opts = {
            "title": "Visualization of mask distance matrix",
            "columnnames": [m[0] for m in masks],
            "rownames": [m[0] for m in masks],
            
           }
    vis.heatmap(distance_matrix, opts=opts)


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    # Pick a free port
    with socket.socket() as s:
        s.bind(("localhost", 0))
        port = s.getsockname()[1]
        url = "tcp://localhost:" + str(port)
        parsed_args.url = url

    process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
