# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import thumt.utils as utils

from thumt.modules.module import Module
from thumt.modules.affine import Affine


class FeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0,
                 name="feed_forward"):
        super(FeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")

        self.reset_parameters()

    def forward(self, x):
        h = F.relu(self.input_transform(x))
        h = F.dropout(h, self.dropout, self.training)
        return self.output_transform(h)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)

class WeightedFeedForward(Module):

    def __init__(self, input_size, hidden_size, num_heads, output_size=None, dropout=0.0, enable_alpha=True, expand_alpha_norm=False, sigmoid_weight=False,
                 name="weighted_feed_forward"):
        super().__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size or input_size
        self.dropout = dropout
        self.enable_alpha = enable_alpha
        self.expand_alpha_norm = expand_alpha_norm
        self.sigmoid_weight = sigmoid_weight
        self.additional_params = []

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")
            if enable_alpha:
                self.alpha = nn.Parameter(torch.empty(num_heads))
                self.add_name(self.alpha, "alpha")
                self.additional_params.append(self.alpha)

        self.reset_parameters()

    def forward(self, x):
        h = F.relu(self.input_transform(x))
        h = F.dropout(h, self.dropout, self.training)
        output = self.output_transform(h)
        if self.enable_alpha:
            if self.sigmoid_weight:
                normalized_alpha = torch.sigmoid(self.alpha)
            else:
                normalized_alpha = F.softmax(self.alpha, dim=0)
                if self.expand_alpha_norm:
                    normalized_alpha = normalized_alpha * self.num_heads 
            output = torch.einsum("n,bnld->bld", normalized_alpha, output)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)

class FitFeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0,
                 name="feed_forward"):
        super(FitFeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")

        self.reset_parameters()

    def prune_dim(self, index):
        self.input_transform = utils.prune_linear_layer(self.input_transform, index, dim=1, scale=False)
        self.output_transform = utils.prune_linear_layer(self.output_transform, index, dim=0, scale=False)
        self.hidden_size = index.size(0)

    def forward(self, x):
        h = F.relu(self.input_transform(x))
        h = F.dropout(h, self.dropout, self.training)
        return self.output_transform(h)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)


class MoEGate(Module):

    def __init__(self, hidden_size, num_experts, topk, name="moe_gate"):
        super(MoEGate, self).__init__(name=name)

        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.topk = topk

        self.experts_balance_loss = 0.

        with utils.scope(name):
            self.gate_transform = Affine(hidden_size, num_experts, bias=False)
            self.gate_noise = Affine(hidden_size, num_experts, bias=False)

        self.reset_parameters()

    def compute_experts_balance_loss(self, weight):
        sum_weight = weight.sum(dim=(0, 1))
        coefficient_of_variance = sum_weight.std() / sum_weight.mean()
        return coefficient_of_variance ** 2

    def forward(self, x):
        weight = self.gate_transform(x) + F.softplus(self.gate_noise(x)) * x.new_zeros(self.num_experts).normal_()
        # average on batch and token
        gates, indexes = weight.mean(dim=(0, 1)).topk(self.topk)
        selected_gates = F.softmax(gates, dim=-1)
        
        _, examplewise_indexes = weight.topk(self.topk)
        select_bool = torch.eye(self.num_experts).to(x).index_select(0, examplewise_indexes.reshape(-1))
        select_bool = select_bool.reshape(examplewise_indexes.size(0), 
                                          examplewise_indexes.size(1), 
                                          self.topk, 
                                          self.num_experts).sum(dim=2)
        weight = F.softmax(weight.masked_fill(select_bool.eq(0), -float("inf")), dim=-1)
        self.experts_balance_loss = self.compute_experts_balance_loss(weight)

        return selected_gates, indexes

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            nn.init.xavier_uniform_(self.gate_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.gate_noise.weight, 2 ** -0.5)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MoEFeedForward(Module):

    def __init__(self, num_experts, topk, input_size, hidden_size, output_size=None, dropout=0.0,
                 name="feed_forward"):
        super(MoEFeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout

        self.num_experts = num_experts
        self.topk = topk

        with utils.scope(name):
            self.moe_gate = MoEGate(input_size, num_experts, topk)

            self.input_transform_weight = nn.Parameter(torch.empty(num_experts, input_size, hidden_size))
            self.input_transform_bias = nn.Parameter(torch.empty(num_experts, hidden_size))
            self.output_transform_weight = nn.Parameter(torch.empty(num_experts, hidden_size, self.output_size))
            self.output_transform_bias = nn.Parameter(torch.empty(num_experts, self.output_size))

            self.add_name(self.input_transform_weight, "input_transform_weight")
            self.add_name(self.input_transform_bias, "input_transform_bias")
            self.add_name(self.output_transform_weight, "output_transform_weight")
            self.add_name(self.output_transform_bias, "output_transform_bias")

        self.reset_parameters()

    def forward(self, x):
        batch_size, seq_len, input_size = x.size(0), x.size(1), x.size(2)
        gates, indexes = self.moe_gate(x)
        input_transform_weight = self.input_transform_weight.index_select(0, indexes.reshape(-1))
        input_transform_weight = input_transform_weight.reshape(self.topk, self.input_size, self.hidden_size)
        input_transform_bias = self.input_transform_bias.index_select(0, indexes.reshape(-1))
        input_transform_bias = input_transform_bias.reshape(self.topk, self.hidden_size)

        output_transform_weight = self.output_transform_weight.index_select(0, indexes.reshape(-1))
        output_transform_weight = output_transform_weight.reshape(self.topk, self.hidden_size, self.output_size)
        output_transform_bias = self.output_transform_bias.index_select(0, indexes.reshape(-1))
        output_transform_bias = output_transform_bias.reshape(self.topk, self.output_size)

        h = F.relu(torch.einsum("kih,bli->blkh", input_transform_weight, x) + input_transform_bias)
        h = F.dropout(h, self.dropout, self.training)
        h = torch.einsum("kho,blkh->blko", output_transform_weight, h) + output_transform_bias

        h = torch.einsum("blko,k->blo", h, gates)
        
        return h

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform_weight)
        nn.init.xavier_uniform_(self.output_transform_weight)
        nn.init.constant_(self.input_transform_bias, 0.0)
        nn.init.constant_(self.output_transform_bias, 0.0)

class PickyFeedForward(Module):

    def __init__(self, input_size, hidden_size, output_size=None, dropout=0.0,
                 weight_function="sigmoid", fake_weight=False,
                 name="feed_forward"):
        super(PickyFeedForward, self).__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.dropout = dropout
        
        self.additional_params = {}

        if weight_function == "sigmoid":
            if fake_weight:
                self.compute_weight = lambda x: torch.sigmoid(x) - torch.sigmoid(x).detach() + torch.ones_like(x)
            else:
                self.compute_weight = torch.sigmoid
        elif weight_function == "softmax":
            if fake_weight:
                self.compute_weight = lambda x: F.softmax(x, 0) - F.softmax(x, 0).detach() + torch.ones_like(x)
            else:
                self.compute_weight = lambda x: F.softmax(x, dim=0)


        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")

        self.reset_parameters()

    def prune_dim(self, index, prune_output=True):
        if len(index["input"]) == 0 and len(index["inter"]) == 0 and len(index["output"]) == 0:
            return
        input_index = utils.reverse_select(index["input"], self.input_transform.weight.size(1))
        inter_index = utils.reverse_select(index["inter"], self.input_transform.weight.size(0))
        self.input_transform = utils.prune_linear_layer(self.input_transform, input_index, dim=1, scale=False)
        self.input_transform = utils.prune_linear_layer(self.input_transform, inter_index, dim=0, scale=True)
        self.output_transform = utils.prune_linear_layer(self.output_transform, inter_index, dim=1, scale=False)

        self.input_size = len(input_index)
        self.hidden_size = len(inter_index)

        if prune_output:
            output_index = utils.reverse_select(index["output"], self.output_transform.weight.size(0))
            self.output_transform = utils.prune_linear_layer(self.output_transform, output_index, dim=0, scale=True)

            self.output_size = len(output_index)

    def reinit_dim(self, index):
        pass
        if len(index["input"]) == 0 and len(index["inter"]) == 0 and len(index["output"]) == 0:
            return
        input_index = utils.reverse_select(index["input"], self.input_transform.weight.size(1))
        inter_index = utils.reverse_select(index["inter"], self.input_transform.weight.size(0))
        output_index = utils.reverse_select(index["output"], self.output_transform.weight.size(0))
        self.input_transform = utils.reinit_linear_layer(self.input_transform, input_index, dim=1)
        self.input_transform = utils.reinit_linear_layer(self.input_transform, inter_index, dim=0)
        self.output_transform = utils.reinit_linear_layer(self.output_transform, inter_index, dim=1)
        self.output_transform = utils.reinit_linear_layer(self.output_transform, output_index, dim=0)

    def forward(self, x):
        # apply soft weights
        if "ffn_input_weight" in self.additional_params.keys():
            input_weight = self.compute_weight(self.additional_params["ffn_input_weight"])
            x = torch.einsum("d,bld->bld", input_weight, x)

        h = F.relu(self.input_transform(x))
        if "ffn_inter_weight" in self.additional_params.keys():
            inter_weight = self.compute_weight(self.additional_params["ffn_inter_weight"])
            h = torch.einsum("d,bld->bld", inter_weight, h)
        h = F.dropout(h, self.dropout, self.training)

        h = self.output_transform(h)
        if "ffn_output_weight" in self.additional_params.keys():
            output_weight = self.compute_weight(self.additional_params["ffn_output_weight"])
            h = torch.einsum("d,bld->bld", output_weight, h)

        return h

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)


