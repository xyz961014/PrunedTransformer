import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from copy import copy, deepcopy
from tqdm import tqdm

import thumt.data as data


def visualize_head_selection(model, pattern, func=None, env=None):
    """
        Visualize head selection in bar using visdom
        model: model to visualize
        pattern: pattern of parameter name
        func: how selection weight is computed, weight = func(related_param, param_name), default by softmax
    """

    if func is None:
        func = lambda var, name: F.softmax(var, dim=0)
    if env is None:
        env = model.name

    import visdom
    vis = visdom.Visdom(env=env)
    assert vis.check_connection()

    for name, var in model.named_parameters():
        if re.search(pattern, name):
            vis.bar(func(var.data, name), win=name, opts={"title": name})


def prune_linear_layer(layer, index: torch.LongTensor, dim: int = 0):
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (:obj:`torch.nn.Linear`): The layer to prune.
        index (:obj:`torch.LongTensor`): The indices to keep in the layer.
        dim (:obj:`int`, `optional`, defaults to 0): The dimension on which to keep the indices.

    Returns:
        :obj:`torch.nn.Linear`: The pruned layer as a new layer with :obj:`requires_grad=True`.
    """
    from thumt.modules import Affine, WeightedAffine
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    if type(layer) == Affine:
        new_layer = Affine(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    elif type(layer) == WeightedAffine:
        new_layer = WeightedAffine(new_size[1] // layer.head_size, 
                                   layer.head_size, 
                                   new_size[0], 
                                   bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer

def prune_vector(vector: torch.nn.Parameter, heads: Set[int], n_heads: int, already_pruned_heads: Set[int]) -> torch.nn.Parameter:
    """
    Prune head weight in pruned head
    """
    mask = torch.ones(n_heads)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long().to(vector.device)
    V = vector.index_select(0, index).clone().detach()

    new_size = len(index)
    new_vector = nn.Parameter(torch.empty(new_size)).to(vector.device)
    new_vector.requires_grad = False
    new_vector.copy_(V.contiguous())
    new_vector.requires_grad = True
    return new_vector

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    """
    Finds the heads and their indices taking :obj:`already_pruned_heads` into account.

    Args:
        heads (:obj:`List[int]`): List of the indices of heads to prune.
        n_heads (:obj:`int`): The number of heads in the model.
        head_size (:obj:`int`): The size of each head.
        already_pruned_heads (:obj:`Set[int]`): A set of already pruned heads.

    Returns:
        :obj:`Tuple[Set[int], torch.LongTensor]`: A tuple with the remaining heads and their corresponding indices.
    """
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def eval_loss(model, dataset, params):
    total_loss = 0.
    num_sentences = 0
    data_len = 0
    for features in dataset:
        data_len += 1
        num_sentences += features[1].shape[0]

    for features in tqdm(dataset, total=data_len):
        features, labels = data.lookup(features, "train", params)
        loss = model(features, labels, mode="eval")
        total_loss += loss.sum().item()
    return total_loss / num_sentences

def head_importance_score(model, method, dataset, sorted_key, eval_dataset, references, params, 
                          visualize=False, env=None, equal_heads=False):
    # the more important, the larger score it gets
    from thumt.utils.evaluation import evaluate
    model.eval()
    if equal_heads:
        # make all heads equal
        model.equal_heads()

    def drop_one_score(score_type="loss"):
        # compute full model score
        with torch.no_grad():
            if score_type == "bleu":
                full_score = evaluate(model, sorted_key, eval_dataset,
                                     params.output, references, params)
            elif score_type == "loss":
                full_score = eval_loss(model, dataset, params)
                print("loss: {:.3f}".format(full_score))
            else:
                raise ValueError("Unkown score type")

            encoder_head_scores = []
            for layer_num, layer in enumerate(model.encoder.layers):
                layer_head_scores = []
                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [head]}
                    copy_model.encoder._prune_heads(head_to_prune)
                    print("Validating model with pruning head {} at encoder layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = score_wo_head - full_score
                    if score_type == "bleu":
                        delta_score = -delta_score
                    layer_head_scores.append(delta_score)
                encoder_head_scores.append(layer_head_scores)

            decoder_head_scores = []
            encdec_head_scores = []
            for layer_num, layer in enumerate(model.decoder.layers):
                decoder_layer_head_scores = []
                encdec_layer_head_scores = []

                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [head]}
                    copy_model.decoder._prune_heads(self_heads_to_prune=head_to_prune, 
                                                    encdec_heads_to_prune={})
                    print("Validating model with pruning head {} at decoder layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = score_wo_head - full_score
                    if score_type == "bleu":
                        delta_score = -delta_score
                    decoder_layer_head_scores.append(delta_score)

                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [head]}
                    copy_model.decoder._prune_heads(self_heads_to_prune={}, 
                                                    encdec_heads_to_prune=head_to_prune)
                    print("Validating model with pruning head {} at encdec layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = score_wo_head - full_score
                    if score_type == "bleu":
                        delta_score = -delta_score
                    encdec_layer_head_scores.append(delta_score)
                decoder_head_scores.append(decoder_layer_head_scores)
                encdec_head_scores.append(encdec_layer_head_scores)

        if visualize:
            visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores)

        return encoder_head_scores, decoder_head_scores, encdec_head_scores

    def confidence():
        encoder_head_scores = None 
        decoder_head_scores = None
        encdec_head_scores = None
        num_sentences = 0
        data_len = 0
        for features in dataset:
            data_len += 1
            num_sentences += features[1].shape[0]

        with torch.no_grad():
            for features in tqdm(dataset, total=data_len):
                features, labels = data.lookup(features, "train", params)
                confidences = model.compute_confidence(features, labels) 
                encoder_confidence, decoder_confidence, encdec_confidence = confidences
                # each confidence score is a list of list, convert to np.array
                encoder_confidence = np.array(encoder_confidence)
                decoder_confidence = np.array(decoder_confidence)
                encdec_confidence = np.array(encdec_confidence)
                if encoder_head_scores is None:
                    encoder_head_scores = encoder_confidence
                else:
                    encoder_head_scores += decoder_confidence
                if decoder_head_scores is None:
                    decoder_head_scores = decoder_confidence
                else:
                    decoder_head_scores += decoder_confidence
                if encdec_head_scores is None:
                    encdec_head_scores = encdec_confidence
                else:
                    encdec_head_scores += encdec_confidence
        
        encoder_head_scores = (encoder_head_scores / num_sentences).tolist()
        decoder_head_scores = (decoder_head_scores / num_sentences).tolist()
        encdec_head_scores = (encdec_head_scores / num_sentences).tolist()
            
        if visualize:
            visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores)

        return encoder_head_scores, decoder_head_scores, encdec_head_scores

    def remain_one_score(score_type="loss"):
        # compute full model score
        with torch.no_grad():
            if score_type == "bleu":
                full_score = evaluate(model, sorted_key, eval_dataset,
                                     params.output, references, params)
            elif score_type == "loss":
                full_score = eval_loss(model, dataset, params)
                print("loss: {:.3f}".format(full_score))
            else:
                raise ValueError("Unkown score type")

            encoder_head_scores = []
            for layer_num, layer in enumerate(model.encoder.layers):
                layer_head_scores = []
                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [h for h in range(params.num_heads) if not h == head]}
                    copy_model.encoder._prune_heads(head_to_prune)
                    print("Validating model with pruning head {} at encoder layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = full_score - score_wo_head
                    if score_type == "bleu":
                        delta_score = -delta_score
                    layer_head_scores.append(delta_score)
                encoder_head_scores.append(layer_head_scores)

            decoder_head_scores = []
            encdec_head_scores = []
            for layer_num, layer in enumerate(model.decoder.layers):
                decoder_layer_head_scores = []
                encdec_layer_head_scores = []

                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [h for h in range(params.num_heads) if not h == head]}
                    copy_model.decoder._prune_heads(self_heads_to_prune=head_to_prune, 
                                                    encdec_heads_to_prune={})
                    print("Validating model with pruning head {} at decoder layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = full_score - score_wo_head
                    if score_type == "bleu":
                        delta_score = -delta_score
                    decoder_layer_head_scores.append(delta_score)

                for head in range(params.num_heads):
                    copy_model = deepcopy(model)
                    head_to_prune = {layer_num: [h for h in range(params.num_heads) if not h == head]}
                    copy_model.decoder._prune_heads(self_heads_to_prune={}, 
                                                    encdec_heads_to_prune=head_to_prune)
                    print("Validating model with pruning head {} at encdec layer {}".format(head, layer_num))
                    if score_type == "bleu":
                        score_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                                params.output, references, params) 
                    elif score_type == "loss":
                        score_wo_head = eval_loss(copy_model, dataset, params)
                        print("loss: {:.3f}".format(score_wo_head))
                    delta_score = full_score - score_wo_head
                    if score_type == "bleu":
                        delta_score = -delta_score
                    encdec_layer_head_scores.append(delta_score)
                decoder_head_scores.append(decoder_layer_head_scores)
                encdec_head_scores.append(encdec_layer_head_scores)

        if visualize:
            visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores)

        return encoder_head_scores, decoder_head_scores, encdec_head_scores



    def grad_sensitivity():
        encoder_head_scores = None 
        decoder_head_scores = None
        encdec_head_scores = None
        num_sentences = 0
        data_len = 0
        for features in dataset:
            data_len += 1
            num_sentences += features[1].shape[0]

        for features in tqdm(dataset, total=data_len):
            model.zero_grad()
            features, labels = data.lookup(features, "train", params)
            scores = model.compute_grad_sensitivity(features, labels) 
            encoder_score, decoder_score, encdec_score = scores
            # each score score is a list of list, convert to np.array
            encoder_score = np.array(encoder_score)
            decoder_score = np.array(decoder_score)
            encdec_score = np.array(encdec_score)
 
            # each score score is a (layer_num * num_heads) Tensor
            if encoder_head_scores is None:
                encoder_head_scores = encoder_score
            else:
                encoder_head_scores += decoder_score
            if decoder_head_scores is None:
                decoder_head_scores = decoder_score
            else:
                decoder_head_scores += decoder_score
            if encdec_head_scores is None:
                encdec_head_scores = encdec_score
            else:
                encdec_head_scores += encdec_score
        
        encoder_head_scores = (encoder_head_scores / num_sentences).tolist()
        decoder_head_scores = (decoder_head_scores / num_sentences).tolist()
        encdec_head_scores = (encdec_head_scores / num_sentences).tolist()
            
        if visualize:
            visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores)

        return encoder_head_scores, decoder_head_scores, encdec_head_scores

    def random_score():
        encoder_head_scores = np.random.rand(len(model.encoder.layers), params.num_heads)
        decoder_head_scores = np.random.rand(len(model.decoder.layers), params.num_heads)
        encdec_head_scores = np.random.rand(len(model.decoder.layers), params.num_heads)
        if visualize:
            visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores)

        return encoder_head_scores, decoder_head_scores, encdec_head_scores


    def visualize_head_scores(encoder_head_scores, decoder_head_scores, encdec_head_scores):
        try:
            import visdom
            vis = visdom.Visdom(env=env)
            assert vis.check_connection()
            
            for layer, scores in enumerate(encoder_head_scores):
                scores = np.concatenate((np.zeros((len(scores), 1)), np.array([scores]).transpose()), axis=1)
                vis.bar(scores, 
                        win="{} encoder_layer{} score".format(method, layer), 
                        opts={
                                "title": "{} encoder_layer{} score".format(method, layer),
                                "stacked": False,
                             })
            for layer, scores in enumerate(decoder_head_scores):
                scores = np.concatenate((np.zeros((len(scores), 1)), np.array([scores]).transpose()), axis=1)
                vis.bar(scores, 
                        win="{} decoder_layer{} score".format(method, layer), 
                        opts={
                                "title": "{} decoder_layer{} score".format(method, layer),
                                "stacked": False,
                             })
            for layer, scores in enumerate(encdec_head_scores):
                scores = np.concatenate((np.zeros((len(scores), 1)), np.array([scores]).transpose()), axis=1)
                vis.bar(scores, 
                        win="{} encdec_layer{} score".format(method, layer), 
                        opts={
                                "title": "{} encdec_layer{} score".format(method, layer),
                                "stacked": False,
                             })
            #vis.heatmap(np.array(encoder_head_scores), win=method + " encoder_heads", 
            #            opts={
            #                    "title": method + " encoder_head_scores",
            #                    "rownames": ["layer{}".format(l) for l in range(len(model.encoder.layers))],
            #                    "columnnames": ["head{}".format(h) for h in range(params.num_heads)]
            #                 })
            #vis.heatmap(np.array(decoder_head_scores), win=method + " decoder_heads", 
            #            opts={
            #                    "title": method + " decoder_head_scores",
            #                    "rownames": ["layer{}".format(l) for l in range(len(model.decoder.layers))],
            #                    "columnnames": ["head{}".format(h) for h in range(params.num_heads)]
            #                 })
            #vis.heatmap(np.array(encdec_head_scores), win=method + " encdec_heads", 
            #            opts={
            #                    "title": method + " encdec_head_scores",
            #                    "rownames": ["layer{}".format(l) for l in range(len(model.decoder.layers))],
            #                    "columnnames": ["head{}".format(h) for h in range(params.num_heads)]
            #                 })

        except:
            print("Visdom does not launch correctly.")

    if method == "drop_one_bleu":
        return drop_one_score(score_type="bleu")
    elif method == "drop_one_loss":
        return drop_one_score(score_type="loss")
    elif method == "confidence":
        return confidence()
    elif method == "remain_one_bleu":
        return remain_one_score(score_type="bleu")
    elif method == "remain_one_loss":
        return remain_one_score(score_type="loss")
    elif method == "grad_sensitivity":
        return grad_sensitivity()
    elif method == "random":
        return random_score()
    else:
        raise ValueError("Unkown head score method {}".format(method))
