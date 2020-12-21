import re
import torch
import torch.nn as nn
import torch.nn.functional as F

def visualize_head_selection(model, pattern, func=None, env=None):
    """
        Visualize head selection in bar using visdom
        model: model to visualize
        pattern: pattern of parameter name
        func: how selection weight is computed, weight = func(related_param), default by softmax
    """

    if func is None:
        func = lambda x: F.softmax(x, dim=0)
    if env is None:
        env = model.name

    import visdom
    vis = visdom.Visdom(env=env)
    assert vis.check_connection()

    for name, var in model.named_parameters():
        if re.search(pattern, name):
            vis.bar(func(var.data), win=name, opts={"title": name})

