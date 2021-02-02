# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numbers
import torch
import torch.nn as nn
import thumt.utils as utils

from thumt.modules.module import Module


class LayerNorm(Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 name="layer_norm"):
        super(LayerNorm, self).__init__(name=name)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        with utils.scope(name):
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
                self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
                self.add_name(self.weight, "weight")
                self.add_name(self.bias, "bias")
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)


class FitLayerNorm(Module):

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True,
                 name="layer_norm"):
        super(FitLayerNorm, self).__init__(name=name)
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        with utils.scope(name):
            if self.elementwise_affine:
                self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
                self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
                self.add_name(self.weight, "weight")
                self.add_name(self.bias, "bias")
            else:
                self.register_parameter('weight', None)
                self.register_parameter('bias', None)
        self.reset_parameters()

    def prune_dim(self, index):
        if not isinstance(index, torch.Tensor):
            index = torch.Tensor(index).long()
        index = index.to(self.weight.device)
        new_shape = (index.size(0),)
        self.normalized_shape = new_shape
        if self.weight is not None:
            W = self.weight.index_select(0, index).clone().detach()
            with utils.scope(self.name):
                self.weight = nn.Parameter(torch.Tensor(*new_shape))
                self.add_name(self.weight, "weight")
            self.weight.requires_grad = False
            self.weight.copy_(W.contiguous())
            self.weight.requires_grad = True
        if self.bias is not None:
            b = self.bias.index_select(0, index).clone().detach()
            with utils.scope(self.name):
                self.bias = nn.Parameter(torch.Tensor(*new_shape))
                self.add_name(self.bias, "bias")
            self.bias.requires_grad = False
            self.bias.copy_(b.contiguous())
            self.bias.requires_grad = True


    def reset_parameters(self):
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        return nn.functional.layer_norm(
            input, self.normalized_shape, self.weight, self.bias, self.eps)

    def extra_repr(self):
        return '{normalized_shape}, eps={eps}, ' \
            'elementwise_affine={elementwise_affine}'.format(**self.__dict__)
