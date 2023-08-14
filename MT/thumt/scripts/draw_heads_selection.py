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
        description="Draw heads selection.",
        usage="draw_heads_selection.py [<args>] [-h | --help]"
    )

    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path to trained checkpoints.")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="num of heads per layer")
    parser.add_argument("--env", type=str, default="",
                        help="env for visdom")

    return parser.parse_args()

def dict_to_matrix(d, length=8):
    lines = []
    for name, layers in d.items():
        for num, layer in layers.items():
            line = []
            for head in range(length):
                if head in layer:
                    line.append(1)
                else:
                    line.append(0)
            lines.append(line)

    return torch.Tensor(lines)

def main(args):

    if not args.env:
        env_name = re.sub("-|/", "", args.checkpoints[0])
    else:
        env_name = args.env

    import visdom
    vis = visdom.Visdom(env=env_name)
    assert vis.check_connection()

    choice_matrixs = []
    for ckp in args.checkpoints:
        choice = json.load(open(ckp, "rb"))
        choice_matrix = dict_to_matrix(choice, length=args.num_heads)
        choice_matrixs.append(choice_matrix.unsqueeze(0))

    choice_matrix = torch.cat(choice_matrixs, dim=0).mean(dim=0)


    rownames = []
    for key in choice.keys():
        for layer in choice[key].keys():
            rownames.append(key + layer)
    opts = {
            "title": "Visualization of mask choice matrix",
            "columnnames": list(range(args.num_heads)),
            "rownames": rownames,
           }
    vis.heatmap(choice_matrix, opts=opts)
   


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
