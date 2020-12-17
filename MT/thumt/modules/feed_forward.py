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

    def __init__(self, input_size, hidden_size, num_heads, output_size=None, dropout=0.0, enable_alpha=True,
                 name="weighted_feed_forward"):
        super().__init__(name=name)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.output_size = output_size or input_size
        self.dropout = dropout
        self.enable_alpha = enable_alpha

        with utils.scope(name):
            self.input_transform = Affine(input_size, hidden_size,
                                          name="input_transform")
            self.output_transform = Affine(hidden_size, self.output_size,
                                           name="output_transform")
            if enable_alpha:
                self.alpha = nn.Parameter(torch.empty(num_heads))
                self.add_name(self.alpha, "alpha")

        self.reset_parameters()

    def forward(self, x):
        h = F.relu(self.input_transform(x))
        h = F.dropout(h, self.dropout, self.training)
        output = self.output_transform(h)
        if self.enable_alpha:
            normalized_alpha = F.softmax(self.alpha, dim=0)
            output = torch.einsum("n,bnld->bld", normalized_alpha, output)
        return output

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.xavier_uniform_(self.output_transform.weight)
        nn.init.constant_(self.input_transform.bias, 0.0)
        nn.init.constant_(self.output_transform.bias, 0.0)
