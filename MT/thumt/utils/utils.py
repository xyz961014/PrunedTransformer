import torch
import torch.nn as nn

def param_in(p, params):
    for param in params:
        if p.equal(param):
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
 
