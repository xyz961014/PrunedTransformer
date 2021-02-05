# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.weighted_transformer
import thumt.models.selective_transformer
import thumt.models.fit_transformer
import thumt.models.thin_transformer
import thumt.models.headwise_transformer
import thumt.models.moe_transformer
import thumt.models.picky_transformer
import thumt.models.residual_transformer


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "weighted_transformer":
        return thumt.models.weighted_transformer.WeightedTransformer
    elif name == "selective_transformer":
        return thumt.models.selective_transformer.SelectiveTransformer
    elif name == "fit_transformer":
        return thumt.models.fit_transformer.FitTransformer
    elif name == "thin_transformer":
        return thumt.models.thin_transformer.ThinTransformer
    elif name == "headwise_transformer":
        return thumt.models.headwise_transformer.HeadWiseTransformer
    elif name == "moe_transformer":
        return thumt.models.moe_transformer.MoETransformer
    elif name == "picky_transformer":
        return thumt.models.picky_transformer.PickyTransformer
    elif name == "residual_transformer":
        return thumt.models.residual_transformer.ResidualTransformer
    else:
        raise LookupError("Unknown model %s" % name)
