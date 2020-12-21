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
import torch

import thumt.data as data
import thumt.models as models
import thumt.utils as utils
from thumt.utils import visualize_head_selection


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze Attention Heads in model with pre-trained checkpoints.",
        usage="analyze_model_head.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, required=True, nargs=2,
                        help="Path to input file.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained checkpoints.")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path to source and target vocabulary.")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the models.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")

    # analyze function
    parser.add_argument("--function", type=str, default="visualize_head_selection", 
                        choices=["visualize_head_selection"],
                        help="Attention Head analyze function")
    parser.add_argument("--pattern", type=str, default="alpha|kappa",
                        help="pattern to find related parameter in visualize_head_selection")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        device_list=[0],
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        decode_batch_size=32,
    )

    return params


def merge_params(params1, params2):
    params = utils.HParams()

    for (k, v) in six.iteritems(params1.values()):
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in six.iteritems(params2.values()):
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not os.path.exists(m_name):
        return params

    with open(m_name) as fd:
        logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_params(params, args):
    params.parse(args.parameters.lower())

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(args.vocabulary[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(args.vocabulary[1])

    params.vocabulary = {
        "source": src_vocab, "target": tgt_vocab
    }
    params.lookup = {
        "source": src_w2idx, "target": tgt_w2idx
    }
    params.mapping = {
        "source": src_idx2w, "target": tgt_idx2w
    }

    return params


def main(args):
    # Load configs
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)

    checkpoint = utils.latest_checkpoint(args.checkpoint)
    env_name = re.sub("-", "", checkpoint).split("/")[-1].split(".")[0]

    torch.set_default_tensor_type(torch.FloatTensor)

    # Create model
    with torch.no_grad():

        model = model_cls(params)

        model.eval()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])

        if args.function == "visualize_head_selection":
            visualize_head_selection(model, args.pattern, env=env_name)


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()
    process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
