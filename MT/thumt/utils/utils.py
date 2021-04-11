import torch
import torch.nn as nn

def param_in(p, params):
    for param in params:
        if id(p) == id(param):
            return True
    else:
        return False

def get_reg_loss(name):
    name = name.lower()
    REG_LOSS_DICT = {
            "l1": nn.L1Loss(reduction="sum"),
            "smoothl1": nn.SmoothL1Loss(beta=0.1, reduction="sum"),
            "l2": nn.MSELoss(reduction="sum")
                    }
    return REG_LOSS_DICT[name]
 
def dim_dropout(x, p=0.5, training=True, dim=0):
    """
    dropout on specified dimension
    """
    if not training or not p:
        return x
    if dim < 0:
        dim = x.dim() + dim
    x = x.clone()
    mask_shape = [x.size(i) if i == dim else 1 for i in range(x.dim())]
    mask = x.new_empty(mask_shape, requires_grad=False).bernoulli_(1 - p)
    mask = mask.div_(1 - p)
    mask = mask.expand_as(x)
    return x * mask

def compute_common_score(compare_list):
    score_tensor = torch.zeros_like(compare_list[0]).bool()
    for i in range(len(compare_list) - 1):
        temp_score = torch.logical_xor(compare_list[i], compare_list[i+1])
        score_tensor = torch.logical_or(score_tensor, temp_score)
    return score_tensor.eq(False).sum().item()

