# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import thumt.utils as utils

from thumt.modules.module import Module

class ConcreteGate(Module):
    """
    Pytorch version of ConcreteGate from the work of Voita et al., 2019
    We delete l2 part of it

    A gate made of stretched concrete distribution (using experimental Stretchable Concrete™)
    Can be applied to sparsify neural network activations or weights.
    Example usage: https://gist.github.com/justheuristic/1118a14a798b2b6d47789f7e6f511abd
    :param shape: shape of gate variable. can be broadcasted.
        e.g. if you want to apply gate to tensor [batch, length, units] over units axis,
        your shape should be [1, 1, units]
    :param temperature: concrete sigmoid temperature, should be in (0, 1] range
        lower values yield better approximation to actual discrete gate but train longer
    :param stretch_limits: min and max value of gate before it is clipped to [0, 1]
        min value should be negative in order to compute l0 penalty as in https://arxiv.org/pdf/1712.01312.pdf
        however, you can also use tf.nn.sigmoid(log_a) as regularizer if min, max = 0, 1
    :param l0_penalty: coefficient on the regularizer that minimizes l0 norm of gated value
    :param eps: a small additive value used to avoid NaNs
    :param hard: if True, gates are binarized to {0, 1} but backprop is still performed as if they were concrete
    :param local_rep: if True, samples a different gumbel noise tensor for each sample in batch,
        by default, noise is sampled using shape param as size.

    """

    def __init__(self, shape, temperature=0.33, stretch_limits=(-0.1, 1.1), 
                 l0_penalty=0.0, eps=1e-6, hard=False, local_rep=False,
                 name="concrete_gate"):
        super().__init__(name=name)

        self.temperature = temperature
        self.stretch_limits = stretch_limits
        self.eps = eps
        self.hard = hard
        self.l0_penalty = l0_penalty
        self.local_rep = local_rep

        with utils.scope(name):
            self.log_a = nn.Parameter(torch.empty(*shape))
            self.add_name(self.log_a, "log_a")

        self.reset_parameters()

    def forward(self, values):
        # apply gates to values
        gates = self.get_gates(shape=values.shape if self.local_rep else None)
        return values * gates

    def get_gates(self, is_training=None, shape=None):
        low, high = self.stretch_limits
        is_training = self.training if is_training is None else is_training

        if is_training:
            shape = self.log_a.shape if shape is None else shape
            noise = torch.empty(*shape)
            noise.uniform_(self.eps, 1.0 - self.eps)
            concrete = torch.sigmoid((torch.log(noise) - torch.log(1 - noise) + self.log_a) / self.temperature)
        else:
            concrete = torch.sigmoid(self.log_a)

        stretched_concrete = concrete * (high - low) + low
        clipped_concrete = stretched_concrete.clamp(0, 1)

        if self.hard:
            hard_concrete = clipped_concrete.ge(0.5).to(clipped_concrete)
            clipped_concrete = clipped_concrete + hard_concrete.clone().detach() - clipped_concrete.clone().detach()

        return clipped_concrete

    def get_penalty(self, dim=[]):
        """
        Computes l0 penalties. 
        (usually activations or weights) before they are multiplied by the gate
        Returns the regularizer value that should to be MINIMIZED (negative logprior)
        """

        #if self.l0_penalty == 0:
        #    print("get_penalty() is called with penaltiy set to 0")

        low, high = self.stretch_limits
        assert low < 0.0, "p_gate_closed can be computed only if lower strech limit is negative"

        p_open = torch.sigmoid(self.log_a - self.temperature * math.log(-low / high))
        p_open = p_open.clamp(self.eps, 1.0 - self.eps)

        total_reg = 0.
        if self.l0_penalty:
            l0_reg = self.l0_penalty * p_open.sum(dim=dim)
            total_reg += l0_reg.mean() 

        return total_reg

    @property
    def sparsity_rate(self):
        is_nonzero = self.get_gates().ne(0.0).float()
        return is_nonzero.mean()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.log_a)
