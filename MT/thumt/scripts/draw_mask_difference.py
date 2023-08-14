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

    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path to trained checkpoints.")
    parser.add_argument("--common_n", type=int, default=5,
                        help="stat common heads in n checkpoint")
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

def compute_mask_difference(masks, i):
    if i == 0:
        return 1
    
    mask_prev = masks[i-1][1]
    mask = masks[i][1]
    return compute_mask_distance(mask_prev, mask)

def compute_common_n_value(masks, n, i):
    if i + 1 < n:
        return 0
    compare_list = [m[1] for m in masks[i+1-n:i+1]]
    return compute_common_score(compare_list)

def compute_common_score(compare_list):
    score_tensor = torch.zeros_like(compare_list[0]).bool()
    for i in range(len(compare_list) - 1):
        temp_score = torch.logical_xor(compare_list[i], compare_list[i+1])
        score_tensor = torch.logical_or(score_tensor, temp_score)
    return score_tensor.eq(False).sum().item()


def main(args):

    if not args.env:
        env_name = re.sub("-|/", "", args.checkpoints[0])
    else:
        env_name = args.env

    import visdom
    vis = visdom.Visdom(env=env_name)
    assert vis.check_connection()

    distance_matrixs = []
    for ckp in args.checkpoints:
        masks = pickle.load(open(ckp, "rb"))
        distance_matrix = compute_distance_matrix(masks)
        distance_matrixs.append(distance_matrix.unsqueeze(0))

    distance_matrix = torch.cat(distance_matrixs, dim=0).mean(dim=0)


    opts = {
            "title": "Visualization of mask distance matrix",
            "columnnames": [m[0] for m in masks],
            "rownames": [m[0] for m in masks],
           }
    vis.heatmap(distance_matrix, opts=opts)

    #steps = np.array([m[0] for m in masks])
    #mask_diffs = np.array([compute_mask_difference(masks, i) for i in range(len(masks))])
    #common_n_values = np.array([compute_common_n_value(masks, args.common_n, i) for i in range(len(masks))])

    #opts_diff = {
    #        "title": "Neighbor mask differences",
    #        "xlabel": "Steps",
    #        "ylabel": "Score"
    #       }
    #opts_common = {
    #        "title": "recent {} masks shared heads".format(args.common_n),
    #        "xlabel": "Steps",
    #        "ylabel": "Heads"
    #       }
    #vis.line(mask_diffs[1:], steps[1:], opts=opts_diff)
    #vis.line(common_n_values[args.common_n:], steps[args.common_n:], opts=opts_common)
    


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
