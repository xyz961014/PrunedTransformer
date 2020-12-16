# coding=utf-8
# Copyright 2017-2020 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.transformer
import thumt.models.weighted_transformer


def get_model(name):
    name = name.lower()

    if name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "weighted_transformer":
        return thumt.models.weighted_transformer.WeightedTransformer
    else:
        raise LookupError("Unknown model %s" % name)
