# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datetime
import glob
import operator
import os
import shutil
import time
import math
import torch

import torch.distributed as dist

from thumt.data.vocab import lookup
from thumt.utils.checkpoint import save, latest_checkpoint
from thumt.utils.inference import beam_search
from thumt.utils.bleu import bleu
from thumt.utils.bpe import BPE
from thumt.utils.misc import get_global_step, get_global_time
from thumt.utils.summary import scalar

from tqdm import tqdm


def _save_log(filename, result):
    metric, global_step, score = result

    with open(filename, "a") as fd:
        time = datetime.datetime.now()
        msg = "%s: %s at step %d: %f\n" % (time, metric, global_step, score)
        fd.write(msg)


def _read_score_record(filename):
    # "checkpoint_name": score
    records = []

    if not os.path.exists(filename):
        return records

    with open(filename) as fd:
        for line in fd:
            name, score = line.strip().split(":")
            name = name.strip()[1:-1]
            score = float(score)
            records.append([name, score])

    return records


def _save_score_record(filename, records):
    keys = []

    for record in records:
        checkpoint_name = record[0]
        step = int(checkpoint_name.strip().split("-")[-1].rstrip(".pt"))
        keys.append((step, record))

    sorted_keys = sorted(keys, key=operator.itemgetter(0),
                         reverse=True)
    sorted_records = [item[1] for item in sorted_keys]

    with open(filename, "w") as fd:
        for record in sorted_records:
            checkpoint_name, score = record
            fd.write("\"%s\": %f\n" % (checkpoint_name, score))


def _add_to_record(records, record, max_to_keep):
    added = None
    removed = None
    models = {}

    for (name, score) in records:
        models[name] = score

    if len(records) < max_to_keep:
        if record[0] not in models:
            added = record[0]
            records.append(record)
    else:
        sorted_records = sorted(records, key=lambda x: -x[1])
        worst_score = sorted_records[-1][1]
        current_score = record[1]

        if current_score >= worst_score:
            if record[0] not in models:
                added = record[0]
                removed = sorted_records[-1][0]
                records = sorted_records[:-1] + [record]

    # Sort
    records = sorted(records, key=lambda x: -x[1])

    return added, removed, records


def _convert_to_string(tensor, params):
    ids = tensor.tolist()

    output = []
    
    eos_id = params.lookup["target"][params.eos.encode("utf-8")]

    for wid in ids:
        if wid == eos_id:
            break
        output.append(params.mapping["target"][wid])

    output = b" ".join(output)

    return output


def _evaluate_model(model, sorted_key, dataset, references, params):
    # Create model
    with torch.no_grad():
        model.eval()

        iterator = iter(dataset)
        counter = 0
        pad_max = 1024

        # count eval dataset
        total_len = 0
        for _ in iterator:
            total_len += 1
        iterator = iter(dataset)

        # Buffers for synchronization
        size = torch.zeros([dist.get_world_size()]).long()
        t_list = [torch.empty([params.decode_batch_size, pad_max]).long()
                  for _ in range(dist.get_world_size())]
        results = []

        if dist.get_rank() == 0:
            pbar = tqdm(total=total_len)
            pbar.set_description("Validating model")

        while True:
            try:
                features = next(iterator)
                features = lookup(features, "infer", params)
                batch_size = features["source"].shape[0]
            except:
                features = {
                    "source": torch.ones([1, 1]).long(),
                    "source_mask": torch.ones([1, 1]).float()
                }
                batch_size = 0

            counter += 1

            # Decode
            seqs, _ = beam_search([model], features, params)

            # Padding
            seqs = torch.squeeze(seqs, dim=1)
            pad_batch = params.decode_batch_size - seqs.shape[0]
            pad_length = pad_max - seqs.shape[1]
            seqs = torch.nn.functional.pad(seqs, (0, pad_length, 0, pad_batch))

            # Synchronization
            size.zero_()
            size[dist.get_rank()].copy_(torch.tensor(batch_size))
            dist.all_reduce(size)
            dist.all_gather(t_list, seqs)

            if size.sum() == 0:
                break

            if dist.get_rank() != 0:
                continue

            for i in range(params.decode_batch_size):
                for j in range(dist.get_world_size()):
                    n = size[j]
                    seq = _convert_to_string(t_list[j][i], params)

                    if i >= n:
                        continue

                    # Restore BPE segmentation
                    seq = BPE.decode(seq)

                    results.append(seq.split())

            if dist.get_rank() == 0:
                pbar.update(1)

    model.train()

    if dist.get_rank() == 0:
        pbar.close()
        restored_results = []

        for idx in range(len(results)):
            restored_results.append(results[sorted_key[idx]])

        return bleu(restored_results, references)
    
    return 0.0


def evaluate(model, sorted_key, dataset, base_dir, references, params):
    if not references:
        return

    base_dir = base_dir.rstrip("/")
    save_path = os.path.join(base_dir, "eval")
    record_name = os.path.join(save_path, "record")
    log_name = os.path.join(save_path, "log")
    max_to_keep = params.keep_top_checkpoint_max

    if dist.get_rank() == 0:
        # Create directory and copy files
        if not os.path.exists(save_path):
            print("Making dir: %s" % save_path)
            os.makedirs(save_path)

            params_pattern = os.path.join(base_dir, "*.json")
            params_files = glob.glob(params_pattern)

            for name in params_files:
                new_name = name.replace(base_dir, save_path)
                shutil.copy(name, new_name)

    # Do validation here
    global_step = get_global_step()
    global_time = get_global_time()

    if dist.get_rank() == 0:
        print("-" * 90)
        print("Validating model at step %d" % global_step)

    score = _evaluate_model(model, sorted_key, dataset, references, params)

    # Save records
    if dist.get_rank() == 0:
        scalar("BLEU/score - step", score, global_step, write_every_n_steps=1)
        scalar("BLEU/score - time", score, math.floor(global_time), write_every_n_steps=1)
        print("BLEU at step %d, training time %d s: %f" % (global_step, math.floor(global_time), score))

        # Save checkpoint to save_path
        save({
                "model": model.state_dict(), 
                "step": global_step,
                "pruned_heads": model.find_pruned_heads() if hasattr(model, "find_pruned_heads") else None
             }, save_path)

        _save_log(log_name, ("BLEU", global_step, score))
        records = _read_score_record(record_name)
        record = [latest_checkpoint(save_path).split("/")[-1], score]

        added, removed, records = _add_to_record(records, record, max_to_keep)

        if added is None:
            # Remove latest checkpoint
            filename = latest_checkpoint(save_path)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        if removed is not None:
            filename = os.path.join(save_path, removed)
            print("Removing %s" % filename)
            files = glob.glob(filename + "*")

            for name in files:
                os.remove(name)

        _save_score_record(record_name, records)

        best_score = records[0][1]
        print("Best score at step %d: %f" % (global_step, best_score))
        print("-" * 90)

    return score
