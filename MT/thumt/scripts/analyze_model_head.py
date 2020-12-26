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
import torch
import torch.distributed as dist
import torch.nn.functional as F

import thumt.data as data
import thumt.models as models
import thumt.utils as utils


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
                        choices=["visualize_head_selection", "head_importance_score"],
                        help="Attention Head analyze function")
    parser.add_argument("--pattern", type=str, default="alpha|kappa",
                        help="pattern to find related parameter in visualize_head_selection")
    parser.add_argument("--head_importance_method", type=str, default="drop_one_loss",
                        choices=["drop_one_bleu", "drop_one_loss", "confidence", "remain_one_loss", "remain_one_bleu", "grad_sensitivity"],
                        help="method to evaluate head importance in head_importance_score")
    parser.add_argument("--equal_heads", action="store_true", 
                        help="make all head equal in head_importance_score")
    parser.add_argument("--env", type=str, default="",
                        help="env for visdom")

    return parser.parse_args()


def default_params():
    params = utils.HParams(
        input=["", ""],
        output="",
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
        # evaluate
        keep_top_checkpoint_max=5,
        buffer_size=10000,
        max_length=256,
        min_length=1,
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
    if os.path.isdir(model_dir):
        model_dir = os.path.abspath(model_dir)
    else:
        model_dir = os.path.abspath(os.path.dirname(model_dir))
    p_name = os.path.join(model_dir, "params.json")
    m_name = os.path.join(model_dir, model_name + ".json")

    if os.path.exists(p_name):
        with open(p_name) as fd:
            logging.info("Restoring hyper parameters from %s" % p_name)
            json_str = fd.readline()
            params.parse_json(json_str)

    if os.path.exists(m_name):
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

def load_references(pattern):
    if not pattern:
        return None

    files = glob.glob(pattern)
    references = []

    for name in files:
        ref = []
        with open(name, "rb") as fd:
            for line in fd:
                items = line.strip().split()
                ref.append(items)
        references.append(ref)

    return list(zip(*references))


def main(args):
    # Load configs
    model_cls = models.get_model(args.model)
    params = default_params()
    params = merge_params(params, model_cls.default_params())
    params = import_params(args.checkpoint, args.model, params)
    params = override_params(params, args)

    # Initialize distributed utility
    dist.init_process_group("nccl", init_method=args.url,
                            rank=args.local_rank,
                            world_size=len(params.device_list))
    torch.cuda.set_device(params.device_list[args.local_rank])
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


    dataset = data.get_dataset(args.input, "train", params)
    sorted_key, eval_dataset = data.get_dataset(args.input[0], "infer", params)
    references = load_references(args.input[1])

    if os.path.isdir(args.checkpoint):
        checkpoint = utils.latest_checkpoint(args.checkpoint)
    else:
        checkpoint = args.checkpoint

    if not args.env:
        env_name = re.sub("-|/", "", checkpoint)
    else:
        env_name = args.env

    # Create model
    model = model_cls(params).cuda()

    model.eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu")["model"])

    if args.function == "visualize_head_selection":
        def compute_head_selection_weight(var, name):
            if re.search("kappa", name):
                if params.expand_kappa_norm:
                    return F.softmax(var, dim=0) * params.num_heads
                else:
                    return F.softmax(var, dim=0)
            elif re.search("alpha", name):
                if params.expand_alpha_norm:
                    return F.softmax(var, dim=0) * params.num_heads
                else:
                    return F.softmax(var, dim=0)
            else:
                return F.softmax(var, dim=0)

        utils.visualize_head_selection(model, args.pattern, func=compute_head_selection_weight, env=env_name)

    elif args.function == "head_importance_score":
        head_scores = utils.head_importance_score(model, args.head_importance_method, 
                                                  dataset, sorted_key, eval_dataset, references, params,
                                                  visualize=True, env=env_name, equal_heads=args.equal_heads)


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
