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
import socket
import time
import torch
import textbrewer
from textbrewer import GeneralDistiller
from textbrewer import TrainingConfig, DistillationConfig
from functools import partial

import thumt.data as data
import torch.distributed as dist
import thumt.models as models
import thumt.optimizers as optimizers
import thumt.utils as utils
import thumt.utils.summary as summary


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Train a neural machine translation model "
                    "with knowledge distillation.",
        usage="trainer.py [<args>] [-h | --help]"
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
                        help="Path to pre-trained teacher model checkpoint.")
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training.")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank of this process.")
    parser.add_argument("--half", action="store_true",
                        help="Enable mixed-precision training.")
    parser.add_argument("--log_interval", type=int, default=100,
                        help="display interval of steps")
    # KD settings
    parser.add_argument("--temperature", type=float, default=4.0,
                        help="temperature for distillation")
    parser.add_argument("--kd_weight", type=float, default=1.0,
                        help="init kd loss weight for distillation")
    parser.add_argument("--kd_final_weight", type=float, default=0.0,
                        help="final kd loss weight for distillation")
    parser.add_argument("--hard_label_init_weight", type=float, default=0.0,
                        help="init hard label weight for distillation")
    parser.add_argument("--hard_label_weight", type=float, default=1.0,
                        help="final hard label weight for distillation")
    parser.add_argument("--kd_attention_loss", action="store_true",
                        help="enable attention matrix alignment loss")
    parser.add_argument("--kd_attention_loss_weight", type=float, default=1.0,
                        help="kd attention loss weight for distillation")
    parser.add_argument("--kd_steps", type=int,
                        help="steps when kd weight linearly drop to 0")
    parser.add_argument("--kd_loss_type", type=str, default="ce",
                        help="type of KD loss")

    # hyperparams
    parser.add_argument("--hparam_set", type=str,
                        help="Name of pre-defined hyper-parameter set.")

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
        initial_step=0,
        warmup_steps=4000,
        train_steps=100000,
        update_cycle=1,
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
        device_list=[0],
        # Checkpoint Saving
        keep_checkpoint_max=20,
        keep_top_checkpoint_max=5,
        save_summary=True,
        save_checkpoint_secs=0,
        save_checkpoint_steps=1000,
        # Validation
        eval_steps=2000,
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


def override_params(params, args, is_teacher=False):
    params.model = args.model or params.model
    params.input = args.input or params.input
    params.output = args.output or params.output
    params.vocab = args.vocabulary or params.vocab
    params.validation = args.validation or params.validation
    params.references = args.references or params.references
    if not is_teacher:
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

def prune_state_dict(t_state_dict, s_state_dict):
    for key in t_state_dict.keys():
        t_tensor = t_state_dict[key]
        s_tensor = s_state_dict[key]
        if not t_tensor.dim() == s_tensor.dim():
            raise ValueError("Params dim of match in %s" % key)
        dim = t_tensor.dim()
        for d in range(dim):
            if t_tensor.size(d) >= s_tensor.size(d):
                t_tensor = t_tensor.index_select(d, torch.arange(s_tensor.size(d)).to(t_tensor).long())
            else:
                raise ValueError("student model bigger in params %s" % key)

        s_state_dict[key].copy_(t_tensor)

    return s_state_dict


def print_variables(model, pattern, log=True):
    flags = []

    for (name, var) in model.named_parameters():
        if re.search(pattern, name):
            flags.append(True)
        else:
            flags.append(False)

    weights = {v[0]: v[1] for v in model.named_parameters()}
    total_size = 0

    for name in sorted(list(weights)):
        if re.search(pattern, name):
            v = weights[name]
            total_size += v.nelement()

            if log:
                print("%s %s" % (name.ljust(60), str(list(v.shape)).rjust(15)))

    if log:
        print("Total trainable variables size: %d" % total_size)

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


def save_checkpoint(step, epoch, model, optimizer, params):
    if dist.get_rank() == 0:
        state = {
            "step": step,
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
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


def get_learning_rate_schedule(params):
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


def get_clipper(params):
    if params.clipping.lower() == "none":
        clipper = None
    elif params.clipping.lower() == "adaptive":
        clipper = optimizers.adaptive_clipper(0.95)
    elif params.clipping.lower() == "global_norm":
        clipper = optimizers.global_norm_clipper(params.clip_grad_norm)
    else:
        raise ValueError("Unknown clipper %s" % params.clipping)

    return clipper


def get_optimizer(params, schedule, clipper):
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


def main(args):
    model_cls = models.get_model(args.model)

    # Import and override parameters
    # Priorities (low -> high):
    # default -> saved -> command
    params = default_params()
    params = merge_params(params, model_cls.default_params(args.hparam_set))
    teacher_params = copy.copy(params)
    teacher_params = import_params(args.checkpoint, args.model, teacher_params)

    teacher_params = override_params(teacher_params, args, is_teacher=True)
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

    teacher_model = model_cls(teacher_params).cuda()
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

    # load teacher model checkpoint
    if args.checkpoint is not None:
        # Load pre-trained models
        t_checkpoint = utils.latest_checkpoint(args.checkpoint)
        t_state = torch.load(t_checkpoint, map_location="cpu")
        teacher_model.load_state_dict(t_state["model"])
        teacher_model.return_state = True
        teacher_model.output_weights = True
        broadcast(teacher_model)
    else:
        raise ValueError("teacher model params not loaded")

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        step = state["step"]
        epoch = state["epoch"]
        model.load_state_dict(state["model"])

        if "optimizer" in state:
            optimizer.load_state_dict(state["optimizer"])
    else:
        step = 0
        epoch = 0

        # initial model with part params of teacher model
        s_state_dict = prune_state_dict(t_state["model"], model.state_dict())
        model.load_state_dict(s_state_dict)

        broadcast(model)

    # KD settings
    if args.kd_steps is None:
        args.kd_steps = params.train_steps

    if args.kd_loss_type == "ce":
        kd_loss_fn = textbrewer.losses.kd_ce_loss
    elif args.kd_loss_type == "mse":
        kd_loss_fn = textbrewer.losses.kd_mse_loss

    def kd_attention_loss_fn(teacher_state, student_state, features):
        if args.kd_loss_type == "ce":
            loss_fn = textbrewer.losses.att_ce_mean_loss
        elif args.kd_loss_type == "mse":
            loss_fn = textbrewer.losses.att_mse_sum_loss

        attention_loss = 0.
        num_matrix = 0
        for t_weight, s_weight in list(zip(teacher_state["encoder_weights"], student_state["encoder_weights"])):
            layer_attn_loss = loss_fn(s_weight, t_weight)
            attention_loss += layer_attn_loss
            num_matrix += 1

        mask = features[0]["target_mask"]
        for t_weight, s_weight in list(zip(teacher_state["decoder_weights"], student_state["decoder_weights"])):
            layer_attn_loss = loss_fn(s_weight, t_weight, mask=mask)
            attention_loss += layer_attn_loss
            num_matrix += 1

        for t_weight, s_weight in list(zip(teacher_state["cross_weights"], student_state["cross_weights"])):
            layer_attn_loss = loss_fn(s_weight, t_weight)
            attention_loss += layer_attn_loss
            num_matrix += 1

        return attention_loss / num_matrix 

    def kd_weight_fn(step):
        if step < args.kd_steps:
            return args.kd_final_weight + \
                   (args.kd_steps - step) / args.kd_steps * (args.kd_weight - args.kd_final_weight)
        else:
            return 0.

    def kd_temp_fn(step):
        if step < args.kd_steps:
            return 1.0 + (args.kd_steps - step) / args.kd_steps * (args.temperature - 1.0)
        else:
            return 1.0

    def hard_weight_fn(step):
        if step < args.kd_steps:
            return args.hard_label_init_weight + \
                   step / args.kd_steps * (args.hard_label_weight - args.hard_label_init_weight)
        else:
            return 1.0

    def train_fn(inputs):
        features, labels = inputs
        state = model(features, labels)
        return state

    def teach_fn(inputs):
        features, labels = inputs
        teacher_model.eval()
        with torch.no_grad():
            _, logits, state = teacher_model(features, labels)
        return logits, state

    counter = 0

    while True:
        start_time = time.time()

        for features in dataset:
            if counter % params.update_cycle == 0:
                step += 1
                utils.set_global_step(step)

            counter += 1
            t = time.time()
            features = data.lookup(features, "train", params)
            hard_label_loss, logits, s_state = train_fn(features)
            # compute KD loss
            soft_label, t_state = teach_fn(features)
            kd_loss = kd_loss_fn(logits, soft_label, kd_temp_fn(step))
            if args.kd_attention_loss:
                kd_attention_loss = kd_attention_loss_fn(t_state, s_state, features)
                kd_loss += kd_attention_loss * args.kd_attention_loss_weight
            kd_weight = kd_weight_fn(step)
            hard_weight = hard_weight_fn(step)
            loss = kd_weight * kd_loss + hard_weight * hard_label_loss
            gradients = optimizer.compute_gradients(loss,
                                                    list(model.parameters()))
            grads_and_vars = exclude_variables(
                trainable_flags,
                zip(gradients, list(model.named_parameters())))
            optimizer.apply_gradients(grads_and_vars)

            t = time.time() - t

            summary.scalar("loss", loss, step, write_every_n_steps=1)
            summary.scalar("global_step/sec", t, step)

            if counter % params.update_cycle == 0:
                if step > 0 and step % args.log_interval == 0 and dist.get_rank() == 0:
                    elapsed = time.time() - start_time
                    if type(optimizer._optimizer._learning_rate) == float:
                        lr = optimizer._optimizer._learning_rate
                    else:
                        lr = optimizer._optimizer._learning_rate(step)
                    print('| epoch {:2d} | step {:8d} | lr {:02.2e} | '
                          'ms/step {:4.0f} | loss {:8.4f} '.format(
                        epoch + 1, step, lr,
                        elapsed * 1000 / args.log_interval, 
                        loss.item()))
                    start_time = time.time()

                if step >= params.train_steps:
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)
                    save_checkpoint(step, epoch, model, optimizer, params)

                    if dist.get_rank() == 0:
                        summary.close()

                    return

                if step % params.eval_steps == 0:
                    utils.evaluate(model, sorted_key, eval_dataset,
                                   params.output, references, params)
                    start_time = time.time()

                if step % params.save_checkpoint_steps == 0:
                    save_checkpoint(step, epoch, model, optimizer, params)
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
