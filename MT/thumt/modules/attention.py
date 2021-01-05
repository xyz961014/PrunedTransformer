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


class Attention(Module):

    def __init__(self, q_size, k_size, hidden_size, name="attention"):
        super(Attention, self).__init__(name)

        self._q_size = q_size
        self._k_size = k_size
        self._hidden_size = hidden_size

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size, name="q_transform")
            self.k_transform = Affine(k_size, hidden_size, name="k_transform")
            self.v_transform = Affine(hidden_size, 1,
                                      name="v_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, 1]
        logits = torch.transpose(logits, 1, 2)
        # [batch, 1, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = torch.softmax(logits, dim=-1)

        # [batch, 1, length]
        weights = torch.squeeze(weights, 2)
        output = torch.matmul(weights, memory)

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight)
            nn.init.xavier_uniform_(self.k_transform.weight)
            nn.init.xavier_uniform_(self.v_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAttentionBase(Module):

    def __init__(self, name="multihead_attention_base"):
        super(MultiHeadAttentionBase, self).__init__(name=name)

    @staticmethod
    def split_heads(x, heads):
        batch = x.shape[0]
        length = x.shape[1]
        channels = x.shape[2]

        y = torch.reshape(x, [batch, length, heads, channels // heads])
        return torch.transpose(y, 2, 1)

    @staticmethod
    def combine_heads(x):
        batch = x.shape[0]
        heads = x.shape[1]
        length = x.shape[2]
        channels = x.shape[3]

        y = torch.transpose(x, 2, 1)

        return torch.reshape(y, [batch, length, heads * channels])


class MultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        head_size = hidden_size // num_heads
        self.head_size = head_size

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")

        self.reset_parameters()

    def forward(self, query, bias, memory=None, kv=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def forward_with_head_analysis(self, query, bias, memory=None, kv=None, mode=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if mode == "confidence":
            head_feature = weights.max(dim=-1)[0].mean(dim=(0, 2))
        elif mode == "grad_sensitivity":
            output.retain_grad()
            head_feature = output
        else:
            raise ValueError("Unknown head analysis mode {}".format(mode))

        if kv is not None:
            return output, k, v, head_feature

        return output, head_feature


    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class MultiHeadAdditiveAttention(MultiHeadAttentionBase):

    def __init__(self, q_size, k_size, hidden_size, num_heads, dropout=0.0,
                 name="multihead_attention"):
        super(MultiHeadAdditiveAttention, self).__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout

        with utils.scope(name):
            self.q_transform = Affine(q_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(k_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, num_heads,
                                      name="v_transform")
            self.o_transform = Affine(k_size, k_size,
                                      name="o_transform")

        self.reset_parameters()

    def compute_cache(self, memory):
        return self.k_transform(memory)

    def forward(self, query, bias, memory, cache=None):
        q = self.q_transform(query)

        if cache is None:
            k = self.k_transform(memory)
        else:
            k = cache

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        # q: [batch, 1, hidden_size]
        # k: [batch, length, hidden_size]
        logits = self.v_transform(torch.tanh(q + k))
        # [batch, length, num_heads]
        logits = torch.transpose(logits, 1, 2)
        # [batch, num_heads, 1, length]
        logits = torch.unsqueeze(logits, 2)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        vh = self.split_heads(memory, self.num_heads)
        x = torch.matmul(weights, vh)

        # combine heads
        output = self.o_transform(self.combine_heads(x))

        return output

    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        elif initializer == "uniform":
            nn.init.uniform_(self.q_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.weight, -0.04, 0.04)
            nn.init.uniform_(self.q_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.k_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.v_transform.bias, -0.04, 0.04)
            nn.init.uniform_(self.o_transform.bias, -0.04, 0.04)
        else:
            raise ValueError("Unknown initializer %d" % initializer)

class WeightedMultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, num_heads, dropout=0.0, enable_kappa=True, expand_kappa_norm=False, sigmoid_weight=False, 
                 name="weighted_multihead_attention"):
        super().__init__(name=name)

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.enable_kappa = enable_kappa
        self.sigmoid_weight = sigmoid_weight
        self.expand_kappa_norm = expand_kappa_norm
        self.additional_params = []

        head_size = hidden_size // num_heads
        self.head_size = head_size

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(hidden_size, hidden_size,
                                      name="o_transform")
            if enable_kappa:
                self.kappa = nn.Parameter(torch.empty(num_heads))
                self.add_name(self.kappa, "kappa")
                self.additional_params.append(self.kappa)


        self.reset_parameters()

    def forward(self, query, bias, memory=None, kv=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)

        if self.enable_kappa:
            # combine kappa weights and combine heads
            if self.sigmoid_weight:
                normalized_kappa = torch.sigmoid(self.kappa)
            else:
                normalized_kappa = F.softmax(self.kappa, dim=0)
                if self.expand_kappa_norm:
                    normalized_kappa = normalized_kappa * self.num_heads
            x = torch.einsum("n,bnld->bnld", normalized_kappa, x)
            output = self.o_transform(self.combine_heads(x))
        else:
            # combine heads
            output = self.o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def forward_with_head_analysis(self, query, bias, memory=None, kv=None, mode=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)

        if self.enable_kappa:
            # combine kappa weights and combine heads
            if self.sigmoid_weight:
                normalized_kappa = torch.sigmoid(self.kappa)
            else:
                normalized_kappa = F.softmax(self.kappa, dim=0)
                if self.expand_kappa_norm:
                    normalized_kappa = normalized_kappa * self.num_heads
            x = torch.einsum("n,bnld->bnld", normalized_kappa, x)
            output = self.o_transform(self.combine_heads(x))
        else:
            # combine heads
            output = self.o_transform(self.combine_heads(x))

        if mode == "confidence":
            head_feature = weights.max(dim=-1)[0].mean(dim=(0, 2))
        elif mode == "grad_sensitivity":
            output.retain_grad()
            head_feature = output
        else:
            raise ValueError("Unknown head analysis mode {}".format(mode))

        if kv is not None:
            return output, k, v, head_feature

        return output, head_feature


    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
            if self.enable_kappa:
                nn.init.constant_(self.kappa, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)


class SelectiveMultiHeadAttention(MultiHeadAttentionBase):

    def __init__(self, hidden_size, head_size, num_heads, dropout=0.0, 
                 input_aware_select=False,
                 select_weight_function="sigmoid",
                 select_method="soft",
                 select_number=0,
                 name="selective_multihead_attention"):
        super().__init__(name=name)

        self.hidden_size = hidden_size
        self.head_size = head_size
        self.num_heads = num_heads
        self.dropout = dropout
        
        self.input_aware_select = input_aware_select
        self.select_method = select_method
        self.select_number = select_number if select_number > 0 else num_heads

        self.additional_params = {}

        attention_hidden_size  = head_size * num_heads
        self.intermediate_size = self.select_number * head_size

        if select_weight_function == "sigmoid":
            self.compute_weight = torch.sigmoid
        elif select_weight_function == "softmax":
            self.compute_weight = lambda x: F.softmax(x, dim=0)

        with utils.scope(name):
            self.q_transform = Affine(hidden_size, attention_hidden_size,
                                      name="q_transform")
            self.k_transform = Affine(hidden_size, attention_hidden_size,
                                      name="k_transform")
            self.v_transform = Affine(hidden_size, attention_hidden_size,
                                      name="v_transform")
            self.o_transform = Affine(attention_hidden_size, hidden_size,
                                      name="o_transform")

        self.reset_parameters()

    def select_head(self, query):
        if self.input_aware_select:
            # compute on batch and token level mean of input
            weights = self.additional_param["select_transform"](query.mean(dim=(0, 1)))
            self.weights = self.compute_weight(weights)
        else:
            self.weights = self.compute_weight(self.additional_params["kappa"])

        if self.select_method == "hard":

            if self.select_number < self.num_heads:
                self.weights, selected_heads = self.weights.topk(k=self.select_number,
                                                                 sorted=False)
                heads = set(h for h in range(self.num_heads))
                selected_heads = set(selected_heads.tolist())
                remove_heads = heads - selected_heads
                remove_heads = list(remove_heads)
            else:
                remove_heads = []
            self.weights = torch.ones_like(self.weights) - self.weights.detach() + self.weights

            _, index = utils.find_pruneable_heads_and_indices(
                remove_heads, 
                self.num_heads, 
                self.head_size, 
                set()
            )
            self.selected_q_transform = utils.selected_linear(self.q_transform, index)
            self.selected_k_transform = utils.selected_linear(self.k_transform, index)
            self.selected_v_transform = utils.selected_linear(self.v_transform, index)
            self.selected_o_transform = utils.selected_linear(self.o_transform, index,
                                                              dim=1)
        elif self.select_method == "soft":
            self.selected_q_transform = lambda x: self.q_transform(x)
            self.selected_k_transform = lambda x: self.k_transform(x)
            self.selected_v_transform = lambda x: self.v_transform(x)
            self.selected_o_transform = lambda x: self.o_transform(x)
        else:
            raise ValueError("Unsupported select method {}".format(self.select_method))

    def forward(self, query, bias, memory=None, kv=None):
        self.select_head(query)

        q = self.selected_q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.selected_k_transform(memory)
            v = v or self.selected_v_transform(memory)
        else:
            # self-attention
            k = self.selected_k_transform(query)
            v = self.selected_v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.select_number)
        kh = self.split_heads(k, self.select_number)
        vh = self.split_heads(v, self.select_number)

        # scale query
        qh = qh * (self.hidden_size // self.select_number) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)
        x = torch.einsum("n,bnld->bnld", self.weights, x)

        # combine heads
        output = self.selected_o_transform(self.combine_heads(x))

        if kv is not None:
            return output, k, v

        return output

    def forward_with_head_analysis(self, query, bias, memory=None, kv=None, mode=None):
        q = self.q_transform(query)

        if memory is not None:
            if kv is not None:
                k, v = kv
            else:
                k, v = None, None

            # encoder-decoder attention
            k = k or self.k_transform(memory)
            v = v or self.v_transform(memory)
        else:
            # self-attention
            k = self.k_transform(query)
            v = self.v_transform(query)

            if kv is not None:
                k = torch.cat([kv[0], k], dim=1)
                v = torch.cat([kv[1], v], dim=1)

        # split heads
        qh = self.split_heads(q, self.num_heads)
        kh = self.split_heads(k, self.num_heads)
        vh = self.split_heads(v, self.num_heads)

        # scale query
        qh = qh * (self.hidden_size // self.num_heads) ** -0.5

        # dot-product attention
        kh = torch.transpose(kh, -2, -1)
        logits = torch.matmul(qh, kh)

        if bias is not None:
            logits = logits + bias

        weights = F.dropout(torch.softmax(logits, dim=-1),
                            p=self.dropout,
                            training=self.training)

        x = torch.matmul(weights, vh)
        # combine heads
        output = self.o_transform(self.combine_heads(x))

        if mode == "confidence":
            head_feature = weights.max(dim=-1)[0].mean(dim=(0, 2))
        elif mode == "grad_sensitivity":
            output.retain_grad()
            head_feature = output
        else:
            raise ValueError("Unknown head analysis mode {}".format(mode))

        if kv is not None:
            return output, k, v, head_feature

        return output, head_feature


    def reset_parameters(self, initializer="uniform_scaling", **kwargs):
        if initializer == "uniform_scaling":
            # 6 / (4 * hidden_size) -> 6 / (2 * hidden_size)
            nn.init.xavier_uniform_(self.q_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.k_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.v_transform.weight, 2 ** -0.5)
            nn.init.xavier_uniform_(self.o_transform.weight)
            nn.init.constant_(self.q_transform.bias, 0.0)
            nn.init.constant_(self.k_transform.bias, 0.0)
            nn.init.constant_(self.v_transform.bias, 0.0)
            nn.init.constant_(self.o_transform.bias, 0.0)
        else:
            raise ValueError("Unknown initializer %d" % initializer)



