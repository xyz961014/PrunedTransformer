import torch
import torch.nn as nn

def param_in(p, params):
    for param in params:
        if p.equal(param):
            return True
    else:
        return False

def get_reg_loss(name):
    REG_LOSS_DICT = {
            "l1": nn.L1Loss(),
            "smoothl1": nn.SmoothL1Loss(),
            "l2": nn.MSELoss()
                    }
    return REG_LOSS_DICT[name]
 
