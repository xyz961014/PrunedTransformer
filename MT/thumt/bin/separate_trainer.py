# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import copy
import glob
import logging
import os
import re
import six
import json
import socket
import pickle
import time
import torch
from pprint import pprint

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.optimizers as optimizers
import thumt.utils as utils
import thumt.utils.summary as summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model. Use an independent optimizer "
                    "to handle additional parameters in evolved transformer",
        usage="separate trainer.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--input", type=str, nargs=2,
                        help="Path to source and target corpus.")
    parser.add_argument("--output", type=str, default="train",
                        help="Path to load/store checkpoints.")
    parser.add_argument("--vocabulary", type=str, nargs=2,
                        help="Path to source and target vocabulary.")
    parser.add_argument("--validation", type=str,
                        help="Path to validation file.")
    parser.add_argument("--references", type=str,
                        help="Pattern to reference files.")
    # settings
    parser.add_argument("--checkpoint", type=str,
                        help="Path to pre-trained checkpoint.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process.")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="display interval of steps")
    parser.add_argument("--display_weights", action="store_true",
                        help="display weights at display step")
    # hyperparams
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper-parameter set.")

    # manually prune or weight
    parser.add_argument("--prune_json", type=str, default="",
                        help="json file containing heads to prune")
    parser.add_argument("--prune_checkpoint", type=str, default="",
                        help="model checkpoint containing heads to prune, will only load which head to prune")
    parser.add_argument("--weight_npy", type=str, default="",
                        help="npy file containing head weights")
    parser.add_argument("--dim_prune_prob", type=float, default=0.0,
                        help="prune dims in FitTransformer or PickyTransformer")
    parser.add_argument("--dim_prune_interval", type=int, default=-1,
                        help="prune dims every N steps in FitTransformer or PickyTransformer"
                             " set to -1 to disable it. set to 0 to only prune at first")
    parser.add_argument("--dim_prune_steps", type=int, nargs="+", default=[],
                        help="list prune steps")
    parser.add_argument("--random_prune", action="store_true",
                        help="random prune heads")
    parser.add_argument("--layerwise", action="store_true",
                        help="Prune heads layerwise")
    parser.add_argument("--check_and_prune", action="store_true",
                        help="prune heads after check mask")
    parser.add_argument("--mask_common", type=int, default=5,
                        help="common choices in recent masks")
    parser.add_argument("--common_score_threshold", type=int, default=0,
                        help="set > 0 use prune when common_score > threshold")

    # model and configuration
    parser.add_argument("--model", type=str, required=True,
                        help="Name of the model.")
    parser.add_argument("--parameters", type=str, default="",
                        help="Additional hyper-parameters.")

    return parser.parse_args(args)


def default_params():
    params = utils.HParams(
        input=["", ""],
        output="",
        model="transformer",
        vocab=["", ""],
        pad="<pad>",
        bos="<eos>",
        eos="<eos>",
        unk="<unk>",
        # Dataset
        batch_size=4096,
        fixed_batch_size=False,
        min_length=1,
        max_length=256,
        buffer_size=10000,
        # Initialization
        initializer_gain=1.0,
        initializer="uniform_unit_scaling",
        # Regularization
        scale_l1=0.0,
        scale_l2=0.0,
        # Training
        update_cycle=1,
        device_list=[0],
        initial_step=0,
        start_step=0,
        warmup_steps=4000,
        train_steps=100000,
        optimizer="Adam",
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        adadelta_rho=0.95,
        adadelta_epsilon=1e-7,
        pattern="",
        clipping="global_norm",
        clip_grad_norm=5.0,
        learning_rate=1.0,
        initial_learning_rate=0.0,
        learning_rate_schedule="linear_warmup_rsqrt_decay",
        learning_rate_boundaries=[0],
        learning_rate_values=[0.0],
        start_decay_step=0,
        end_decay_step=0,
        # Additional Optimizer 
        additional_initial_step=0,
        additional_start_step=0,
        additional_warmup_steps=4000,
        additional_train_steps=100000,
        additional_optimizer="Adam",
        additional_adam_beta1=0.9,
        additional_adam_beta2=0.999,
        additional_adam_epsilon=1e-8,
        additional_adadelta_rho=0.95,
        additional_adadelta_epsilon=1e-7,
        additional_pattern="",
        additional_clipping="global_norm",
        additional_clip_grad_norm=5.0,
        additional_learning_rate=1.0,
        additional_initial_learning_rate=0.0,
        additional_learning_rate_schedule="constant",
        additional_learning_rate_boundaries=[0],
        additional_learning_rate_values=[0.0],
        additional_start_decay_step=0,
        additional_end_decay_step=0,
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Validation
        eval_steps=2000,
        check_mask_steps=0,
        eval_secs=0,
        top_beams=1,
        beam_size=4,
        decode_batch_size=32,
        decode_alpha=0.6,
        decode_ratio=1.0,
        decode_length=50,
        validation="",
        references="",
    )

    return params


def import_params(model_dir, model_name, params):
    model_dir = os.path.abspath(model_dir)
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


def export_params(output_dir, name, params):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save params as params.json
    filename = os.path.join(output_dir, name)

    with open(filename, "w") as fd:
        fd.write(params.to_json())


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


def override_params(params, args):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    params.parse(args.parameters.lower())

    src_vocab, src_w2idx, src_idx2w = data.load_vocabulary(params.vocab[0])
    tgt_vocab, tgt_w2idx, tgt_idx2w = data.load_vocabulary(params.vocab[1])

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


def collect_params(all_params, params):
    collected = utils.HParams()

    for k in six.iterkeys(params.values()):
        collected.add_hparam(k, getattr(all_params, k))

    return collected


def print_variables(model, pattern, log=True):
    flags = []

    if hasattr(model, "additional_params"):
        exclude_params = model.additional_params
    else:
        exclude_params = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name) and not utils.param_in(var, exclude_params):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters() if not utils.param_in(v[1], exclude_params)}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable model variables size: %d" % total_size)

    return flags

def print_additional_variables(model, pattern, log=True):
    flags = []

    if hasattr(model, "additional_params"):
        additional_params = model.additional_params
    else:
        additional_params = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name) and utils.param_in(var, additional_params):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters() if utils.param_in(v[1], additional_params)}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable additional variables size: %d" % total_size)

    return flags



def exclude_variables(flags, grads_and_vars):
    idx = 0
    new_grads = []
    new_vars = []

    for grad, (name, var) in grads_and_vars:
        if flags[idx]:
            new_grads.append(grad)
            new_vars.append((name, var))

        idx += 1

    return zip(new_grads, new_vars)


def save_checkpoint(step, additional_step, epoch, model, optimizer, binary_masks, params):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "additional_step": additional_step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "pruned_heads": model.find_pruned_heads() if hasattr(model, "find_pruned_heads") else None,
        }
        utils.save(state, params.output, params.keep_checkpoint_max)


def infer_gpu_num(param_str):
    result = re.match(r".*device_list=\[(.*?)\].*", param_str)

    if not result:
        return 1
    else:
        dev_str = result.groups()[-1]
        return len(dev_str.split(","))


def broadcast(model):
    for var in model.parameters():
        dist.broadcast(var.data, 0)


def get_learning_rate_schedule(params, additional=False):
    if additional:
        if params.additional_learning_rate_schedule == "linear_warmup_rsqrt_decay":
            schedule = optimizers.LinearWarmupRsqrtDecay(
                params.additional_learning_rate, params.additional_warmup_steps,
                initial_learning_rate=params.additional_initial_learning_rate,
                summary=params.save_summary)
        elif params.additional_learning_rate_schedule == "piecewise_constant_decay":
            schedule = optimizers.PiecewiseConstantDecay(
                params.additional_learning_rate_boundaries, params.additional_learning_rate_values,
                summary=params.save_summary)
        elif params.additional_learning_rate_schedule == "linear_exponential_decay":
            schedule = optimizers.LinearExponentialDecay(
                params.additional_learning_rate, params.additional_warmup_steps,
                params.additional_start_decay_step, params.additional_end_decay_step,
                dist.get_world_size(), summary=params.save_summary)
        elif params.additional_learning_rate_schedule == "constant":
            schedule = params.additional_learning_rate
        else:
            raise ValueError("Unknown schedule %s" % params.additional_learning_rate_schedule)

    else:
        if params.learning_rate_schedule == "linear_warmup_rsqrt_decay":
            schedule = optimizers.LinearWarmupRsqrtDecay(
                params.learning_rate, params.warmup_steps,
                initial_learning_rate=params.initial_learning_rate,
                summary=params.save_summary)
        elif params.learning_rate_schedule == "piecewise_constant_decay":
            schedule = optimizers.PiecewiseConstantDecay(
                params.learning_rate_boundaries, params.learning_rate_values,
                summary=params.save_summary)
        elif params.learning_rate_schedule == "linear_exponential_decay":
            schedule = optimizers.LinearExponentialDecay(
                params.learning_rate, params.warmup_steps,
                params.start_decay_step, params.end_decay_step,
                dist.get_world_size(), summary=params.save_summary)
        elif params.learning_rate_schedule == "constant":
            schedule = params.learning_rate
        else:
            raise ValueError("Unknown schedule %s" % params.learning_rate_schedule)

    return schedule


def get_clipper(params, additional=False):
    if additional:
        if params.additional_clipping.lower() == "none":
            clipper = None
        elif params.additional_clipping.lower() == "adaptive":
            clipper = optimizers.adaptive_clipper(0.95)
        elif params.additional_clipping.lower() == "global_norm":
            clipper = optimizers.global_norm_clipper(params.additional_clip_grad_norm)
        else:
            raise ValueError("Unknown clipper %s" % params.additional_clipping)
    else:
        if params.clipping.lower() == "none":
            clipper = None
        elif params.clipping.lower() == "adaptive":
            clipper = optimizers.adaptive_clipper(0.95)
        elif params.clipping.lower() == "global_norm":
            clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
        else:
            raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper, additional=False):
    if additional:
        if params.additional_optimizer.lower() == "adam":
            optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                                 beta_1=params.additional_adam_beta1,
                                                 beta_2=params.additional_adam_beta2,
                                                 epsilon=params.additional_adam_epsilon,
                                                 clipper=clipper,
                                                 summaries=params.save_summary)
        elif params.additional_optimizer.lower() == "adadelta":
            optimizer = optimizers.AdadeltaOptimizer(
                learning_rate=schedule, rho=params.additional_adadelta_rho,
                epsilon=params.additional_adadelta_epsilon, clipper=clipper,
                summaries=params.save_summary)
        elif params.additional_optimizer.lower() == "sgd":
            optimizer = optimizers.SGDOptimizer(
                learning_rate=schedule, clipper=clipper,
                summaries=params.save_summary)
        else:
            raise ValueError("Unknown optimizer %s" % params.additional_optimizer)
    else:
        if params.optimizer.lower() == "adam":
            optimizer = optimizers.AdamOptimizer(learning_rate=schedule,
                                                 beta_1=params.adam_beta1,
                                                 beta_2=params.adam_beta2,
                                                 epsilon=params.adam_epsilon,
                                                 clipper=clipper,
                                                 summaries=params.save_summary)
        elif params.optimizer.lower() == "adadelta":
            optimizer = optimizers.AdadeltaOptimizer(
                learning_rate=schedule, rho=params.adadelta_rho,
                epsilon=params.adadelta_epsilon, clipper=clipper,
                summaries=params.save_summary)
        elif params.optimizer.lower() == "sgd":
            optimizer = optimizers.SGDOptimizer(
                learning_rate=schedule, clipper=clipper,
                summaries=params.save_summary)
        else:
            raise ValueError("Unknown optimizer %s" % params.optimizer)

    return optimizer


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

def prune_model(model, optimizer, additional_optimizer, json_file):
    if json_file and os.path.exists(json_file):
        with open(json_file) as fp_json:
            heads_to_prune = json.load(fp_json)
        indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
        optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
        additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
        model.prune_heads(heads_to_prune)
        model.prune_dim(indexes_to_prune)


def main(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    params = import_params(args.output, args.model, params)
    params = override_params(params, args)

    # Initialize distributed utility
    if args.distributed:
        dist.init_process_group("nccl")
        torch.cuda.set_device(args.local_rank)
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    else:
        dist.init_process_group("nccl", init_method=args.url,
                                rank=args.local_rank,
                                world_size=len(params.device_list))
        torch.cuda.set_device(params.device_list[args.local_rank])
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # Export parameters
    if dist.get_rank() == 0:
        export_params(params.output, "params.json", params)
        export_params(params.output, "%s.json" % params.model,
                      collect_params(params, model_cls.default_params()))

    model = model_cls(params).cuda()

    if args.half:
        model = model.half()
        torch.set_default_dtype(torch.half)
        torch.set_default_tensor_type(torch.cuda.HalfTensor)

    model.train()

    # Init tensorboard
    summary.init(params.output, params.save_summary)

    schedule = get_learning_rate_schedule(params)
    clipper = get_clipper(params)
    optimizer = get_optimizer(params, schedule, clipper)

    if args.half:
        optimizer = optimizers.LossScalingOptimizer(optimizer)

    optimizer = optimizers.MultiStepOptimizer(optimizer, params.update_cycle)

    trainable_flags = print_variables(model, params.pattern,
                                      dist.get_rank() == 0)
    # Additional Optimizer

    additional_schedule = get_learning_rate_schedule(params, additional=True)
    additional_clipper = get_clipper(params, additional=True)
    additional_optimizer = get_optimizer(params, additional_schedule, additional_clipper, additional=True)

    if args.half:
        additional_optimizer = optimizers.LossScalingOptimizer(additional_optimizer)

    additional_optimizer = optimizers.MultiStepOptimizer(additional_optimizer, params.update_cycle)

    additional_flags = print_additional_variables(model, params.additional_pattern,
                                                  dist.get_rank() == 0)


    dataset = data.get_dataset(params.input, "train", params)

    if params.validation:
        sorted_key, eval_dataset = data.get_dataset(
            params.validation, "infer", params)
        references = load_references(params.references)
    else:
        sorted_key = None
        eval_dataset = None
        references = None

    # Load checkpoint
    checkpoint = utils.latest_checkpoint(params.output)

    if args.checkpoint is not None:
        # Load pre-trained models
        if os.path.isdir(args.checkpoint):
            state = torch.load(utils.latest_checkpoint(args.checkpoint), map_location="cpu")
        else:
            state = torch.load(args.checkpoint, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(state["model"], strict=False)
        if len(missing_keys) > 0:
            if dist.get_rank() == 0:
                print("Missing Key(s) in state_dict: ")
                for key in missing_keys:
                    print(key)
        if len(unexpected_keys) > 0:
            if dist.get_rank() == 0:
                print("Unexpected Key(s) in state_dict: ")
                for key in unexpected_keys:
                    print(key)
        if args.weight_npy and os.path.exists(args.weight_npy):
            model.load_kappa_weights(args.weight_npy)
        if args.prune_json:
            prune_model(model, optimizer, additional_optimizer, args.prune_json)
            if dist.get_rank() == 0:
                print("Model params after dim prune")
                print_variables(model, params.pattern, dist.get_rank() == 0)
        if args.dim_prune_prob and args.dim_prune_interval == 0:
            if model.name == "fit_transformer":
                model.prune_dim(p=args.dim_prune_prob)
            elif model.name == "picky_transformer":
                if args.prune_checkpoint:
                    heads_to_prune = torch.load(args.prune_checkpoint)["pruned_heads"]
                else:
                    heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, random=args.random_prune)
                indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                model.prune_heads(heads_to_prune)
                model.prune_dim(indexes_to_prune)
                if dist.get_rank() == 0:
                    print("Pruned Heads:")
                    pprint(heads_to_prune)
                    with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                        json.dump(heads_to_prune, fp)
            if dist.get_rank() == 0:
                print("Model params after dim prune")
                print_variables(model, params.pattern, dist.get_rank() == 0)
        step = params.initial_step
        additional_step = params.additional_initial_step
        epoch = 0
        broadcast(model)
    elif checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        additional_step = state["additional_step"]
        epoch = state["epoch"]
        if args.weight_npy and os.path.exists(args.weight_npy):
            model.load_kappa_weights(args.weight_npy)
        if not args.prune_json:
            args.prune_json = os.path.join(params.output, "heads_to_prune.json")
        prune_model(model, optimizer, additional_optimizer, args.prune_json)
        model.load_state_dict(state["model"])
        if args.dim_prune_prob and args.dim_prune_interval == 0 and step == 0:
            if model.name == "fit_transformer":
                model.prune_dim(p=args.dim_prune_prob)
            elif model.name == "picky_transformer":
                if args.prune_checkpoint:
                    heads_to_prune = torch.load(args.prune_checkpoint)["pruned_heads"]
                else:
                    heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, random=args.random_prune)
                indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                model.prune_heads(heads_to_prune)
                model.prune_dim(indexes_to_prune)
                if dist.get_rank() == 0:
                    print("Pruned Heads:")
                    pprint(heads_to_prune)
                    with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                        json.dump(heads_to_prune, fp)
            if dist.get_rank() == 0:
                print("Model params after dim prune")
            print_variables(model, params.pattern, dist.get_rank() == 0)

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        additional_step = 0
        epoch = 0
        if args.weight_npy and os.path.exists(args.weight_npy):
            model.load_kappa_weights(args.weight_npy)
        prune_model(model, optimizer, additional_optimizer, args.prune_json)
        if args.dim_prune_prob and args.dim_prune_interval == 0:
            if model.name == "fit_transformer":
                model.prune_dim(p=args.dim_prune_prob)
            elif model.name == "picky_transformer":
                if args.prune_checkpoint:
                    heads_to_prune = torch.load(args.prune_checkpoint)["pruned_heads"]
                else:
                    heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, random=args.random_prune)
                indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                model.prune_heads(heads_to_prune)
                model.prune_dim(indexes_to_prune)
                if dist.get_rank() == 0:
                    print("Pruned Heads:")
                    pprint(heads_to_prune)
                    with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                        json.dump(heads_to_prune, fp)
            if dist.get_rank() == 0:
                print("Model params after dim prune")
                print_variables(model, params.pattern, dist.get_rank() == 0)
        broadcast(model)

    def train_fn(inputs):
        features, labels = inputs
        loss = model(features, labels)
        return loss

    counter = 0
    binary_masks = []

    check_binary_masks = []
    check_mask = False
    check_step = 0
    global_start_time = time.time()

    while True:
        start_time = time.time()

        for features in dataset:
            if counter % params.update_cycle == 0:
                if not check_mask:
                    step += 1
                    utils.set_global_step(step)
                    if step > params.additional_start_step:
                        additional_step += 1
                else:
                    check_step += 1

            counter += 1
            t = time.time()
            features = data.lookup(features, "train", params)
            loss = train_fn(features)
            gradients = optimizer.compute_gradients(loss,
                                                    list(model.parameters()))

            if True in trainable_flags and params.start_step < step < params.start_step + params.train_steps and not check_mask:
                grads_and_vars = exclude_variables(
                    trainable_flags,
                    zip(gradients, list(model.named_parameters())))
                optimizer.apply_gradients(grads_and_vars)

            if True in additional_flags and params.additional_start_step < step < params.additional_start_step + params.additional_train_steps and not check_mask:
                # update grads for additional optimizer
                additional_grads_and_vars = exclude_variables(
                    additional_flags,
                    zip(gradients, list(model.named_parameters())))
                additional_optimizer.apply_gradients(additional_grads_and_vars)

            # check mask 
            if True in additional_flags and check_mask and check_step <= params.check_mask_steps:
                additional_grads_and_vars = exclude_variables(
                    additional_flags,
                    zip(gradients, list(model.named_parameters())))
                additional_optimizer.apply_gradients(additional_grads_and_vars)

            t = time.time() - t

            summary.scalar("loss", loss, step, write_every_n_steps=1)
            summary.scalar("global_step/sec", t, step)

            if counter % params.update_cycle == 0:
                if not check_mask and hasattr(model, "summary_weights"):
                    model.summary_weights(summary, step)

                if step > 0 and step % args.log_interval == 0 and dist.get_rank() == 0:
                    elapsed = time.time() - start_time
                    if True in trainable_flags and params.start_step < step < params.start_step + params.train_steps and not check_mask:
                        if type(optimizer._optimizer._learning_rate) == float:
                            lr = optimizer._optimizer._learning_rate
                        else:
                            lr = optimizer._optimizer._learning_rate(step)
                        print('| epoch {:2d} | step {:17d} | lr {:02.2e} | '
                              'ms/step {:4.0f} | loss {:8.4f} '.format(
                            epoch + 1, step, lr,
                            elapsed * 1000 / args.log_interval, 
                            loss.item()))
                    if True in additional_flags and (params.additional_start_step < step < params.additional_start_step + params.additional_train_steps and not check_mask):

                        heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, layerwise=args.layerwise)
                        binary_mask = model.get_binary_head_mask(heads_to_prune)
                        if len(binary_masks) > 0 and binary_masks[-1].size(0) == binary_mask.size(0):
                            mask_difference = (binary_masks[-1] - binary_mask).abs().sum()
                        else:
                            mask_difference = binary_mask.size(0)
                        if args.mask_common > 0:
                            if len(binary_masks) >= args.mask_common and binary_masks[-args.mask_common].size(0) == binary_mask.size(0):
                                compare_list = binary_masks[-args.mask_common:]
                                compare_list.append(binary_mask)
                                common_score = utils.compute_common_score(compare_list)
                            else:
                                common_score = 0
                        binary_masks.append(binary_mask)

                        if type(additional_optimizer._optimizer._learning_rate) == float:
                            additional_lr = additional_optimizer._optimizer._learning_rate
                        else:
                            additional_lr = additional_optimizer._optimizer._learning_rate(additional_step)

                        print('| epoch {:2d} | additional step {:6d} | lr {:02.2e} | '
                              'ms/step {:4.0f} | loss {:8.4f} | mask diff {:2d} | common {} {:2d}'.format(
                            epoch + 1, additional_step, additional_lr,
                            elapsed * 1000 / args.log_interval, 
                            loss.item(),
                            mask_difference,
                            args.mask_common, common_score))

                    start_time = time.time()

                if check_mask and check_step <= params.check_mask_steps and dist.get_rank() == 0:
                    heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, layerwise=args.layerwise)
                    binary_mask = model.get_binary_head_mask(heads_to_prune)
                    if len(binary_masks) > 0 and binary_masks[-1].size(0) == binary_mask.size(0):
                        mask_difference = (binary_masks[-1] - binary_mask).abs().sum()
                    else:
                        mask_difference = binary_mask.size(0)
                    if args.mask_common > 0:
                        if len(binary_masks) >= args.mask_common and binary_masks[-args.mask_common].size(0) == binary_mask.size(0):
                            compare_list = binary_masks[-args.mask_common:]
                            compare_list.append(binary_mask)
                            common_score = utils.compute_common_score(compare_list)
                        else:
                            common_score = 0
                    binary_masks.append(binary_mask)

                    if type(additional_optimizer._optimizer._learning_rate) == float:
                        additional_lr = additional_optimizer._optimizer._learning_rate
                    else:
                        additional_lr = additional_optimizer._optimizer._learning_rate(additional_step)

                    if check_step % args.log_interval == 0:
                        print('| epoch {:2d} | check mask step {:3d} | lr {:02.2e} | '
                              'loss {:8.4f} | mask diff {:2d} | common {} {:2d}'.format(
                            epoch + 1, check_step, additional_lr,
                            loss.item(),
                            mask_difference,
                            args.mask_common, common_score))



                if step >= max(params.start_step + params.train_steps, params.additional_start_step + params.additional_train_steps):
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)
                    save_checkpoint(step, additional_step, epoch, model, optimizer, binary_masks, params)

                    if dist.get_rank() == 0:
                        summary.close()

                    return

                if not check_mask and not args.check_and_prune and ((args.dim_prune_interval > 0 and step > 0 and step % args.dim_prune_interval == 0) or step in args.dim_prune_steps):
                    if model.name == "picky_transformer":
                        heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, random=args.random_prune,
                                                                    layerwise=args.layerwise)
                        indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                        optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        model.prune_heads(heads_to_prune)
                        model.prune_dim(indexes_to_prune)
                        if dist.get_rank() == 0:
                            print("Pruned Heads:")
                            pprint(heads_to_prune)
                            with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                                json.dump(heads_to_prune, fp)
                    if dist.get_rank() == 0:
                        print("Model params after prune")
                        print_variables(model, params.pattern, dist.get_rank() == 0)
                        params.check_mask_steps = 0


                if not check_mask and step % params.eval_steps == 0:
                    if dist.get_rank() == 0:
                        training_time = time.time() - global_start_time
                        utils.set_global_time(training_time)
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)

                    if args.display_weights:
                        model.display_weights(step)
                    if params.check_mask_steps > 0:
                        model.reinit_kappa_and_ffn_weights()
                        check_mask = True

                    start_time = time.time()
                
                if check_step > 0 and check_step == params.check_mask_steps:
                    check_step = 0
                    check_mask = False
                    # check mask and save
                    heads_to_prune = model.find_pruneable_heads(args.dim_prune_prob, layerwise=args.layerwise)
                    binary_mask = model.get_binary_head_mask(heads_to_prune)
                    if len(check_binary_masks) > 0 and check_binary_masks[-1].size(0) == binary_mask.size(0):
                        mask_difference = (check_binary_masks[-1] - binary_mask).abs().sum()
                    else:
                        mask_difference = binary_mask.size(0)
                    if args.mask_common > 0:
                        if len(check_binary_masks) >= args.mask_common and check_binary_masks[-args.mask_common].size(0) == binary_mask.size(0):
                            compare_list = check_binary_masks[-args.mask_common:]
                            compare_list.append(binary_mask)
                            common_score = utils.compute_common_score(compare_list)
                        else:
                            common_score = 0
                    check_binary_masks.append(binary_mask)
                    if dist.get_rank() == 0:
                        print("-" * 50)
                        print('| check mask diff {:2d} | common {} {:2d}'.format(
                            mask_difference, args.mask_common, common_score))
                        pickle.dump([((i + 1) * params.eval_steps, mask.cpu()) 
                                     for i, mask in enumerate(check_binary_masks)], 
                                    open(os.path.join(params.output, "check_masks.pkl"), "wb"))
                        if len(check_binary_masks) > args.mask_common > 0 and check_binary_masks[-args.mask_common].size(0) == binary_mask.size(0):
                            common_mask, rest_random_mask = utils.choose_common_mask(compare_list, args.dim_prune_prob)
                            print("common {} Heads:".format(args.mask_common))
                            pprint(model.get_heads_from_mask(common_mask))
                            print("pruned {} Heads:".format(args.mask_common))
                            pprint(model.get_heads_from_mask(binary_mask.masked_fill(common_mask.eq(False), 0)))
                            with open(os.path.join(params.output, "common_heads.json"), "w") as fp:
                                json.dump(model.get_heads_from_mask(common_mask), fp)
                            with open(os.path.join(params.output, "keep_common_random_rest.json"), "w") as fp:
                                json.dump(model.get_heads_from_mask(rest_random_mask), fp)

                    if args.check_and_prune and ((args.dim_prune_interval > 0 and step > 0 and step % args.dim_prune_interval == 0) or step in args.dim_prune_steps):
                        indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                        optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        model.prune_heads(heads_to_prune)
                        model.prune_dim(indexes_to_prune)
                        if dist.get_rank() == 0:
                            print("Pruned Heads:")
                            pprint(heads_to_prune)
                            with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                                json.dump(heads_to_prune, fp)
                        if dist.get_rank() == 0:
                            print("Model params after prune")
                        print_variables(model, params.pattern, dist.get_rank() == 0)

                    if common_score >= args.common_score_threshold > 0:
                        # keep common choice and random pick the rest 
                        common_mask, rest_random_mask = utils.choose_common_mask(compare_list, args.dim_prune_prob)
                        heads_to_prune = model.get_heads_from_mask(rest_random_mask)
                        indexes_to_prune = model.find_pruneable_dim(heads_to_prune)
                        optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        additional_optimizer.prune_heads_and_dims(heads_to_prune, indexes_to_prune, model)
                        model.prune_heads(heads_to_prune)
                        model.prune_dim(indexes_to_prune)
                        if dist.get_rank() == 0:
                            print("Pruned Heads:")
                            pprint(heads_to_prune)
                            with open(os.path.join(params.output, "heads_to_prune.json"), "w") as fp:
                                json.dump(heads_to_prune, fp)
                        if dist.get_rank() == 0:
                            print("Model params after prune")
                        print_variables(model, params.pattern, dist.get_rank() == 0)
                        args.common_score_threshold = 0


                if not check_mask and step % params.save_checkpoint_steps == 0:
                    save_checkpoint(step, additional_step, epoch, model, optimizer, binary_masks, params)
                    pickle.dump([((i + 1) * args.log_interval, mask.cpu()) 
                                 for i, mask in enumerate(binary_masks)], 
                                open(os.path.join(params.output, "masks.pkl"), "wb"))
                    start_time = time.time()

        epoch += 1


# Wrap main function
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)


def cli_main():
    parsed_args = parse_args()

    if parsed_args.distributed:
        main(parsed_args)
    else:
        # Pick a free port
        with socket.socket() as s:
            s.bind(("localhost", 0))
            port = s.getsockname()[1]
            url = "tcp://localhost:" + str(port)
            parsed_args.url = url

        world_size = infer_gpu_num(parsed_args.parameters)

        if world_size > 1:
            torch.multiprocessing.spawn(process_fn, args=(parsed_args,),
                                        nprocs=world_size)
        else:
            process_fn(0, parsed_args)


if __name__ == "__main__":
    cli_main()
