# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.weighted_transformer
import thumt.models.selective_transformer
import thumt.models.fit_transformer


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
    else:
        raise LookupError("Unknown model %s" % name)
