import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from copy import copy, deepcopy


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


def prune_linear_layer(layer: torch.nn.Linear, index: torch.LongTensor, dim: int = 0) -> torch.nn.Linear:
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


def head_importance_score(model, method, sorted_key, eval_dataset, references, params):
    from thumt.utils.evaluation import evaluate
    def drop_one_score():
        #full_bleu = evaluate(model, sorted_key, eval_dataset,
        #                     params.output, references, params)
        full_bleu = 0.1

        encoder_head_scores = []
        for num, layer in enumerate(model.encoder.layers):
            layer_head_scores = []
            for head in range(params.num_heads):
                copy_model = deepcopy(model)
                head_to_prune = {num: [head]}
                copy_model.encoder._prune_heads(head_to_prune)
                bleu_wo_head = evaluate(copy_model, sorted_key, eval_dataset,
                                        params.output, references, params) 
                import ipdb
                ipdb.set_trace()

    if method == "drop_one":
        return drop_one_score()
