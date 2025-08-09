import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize(x, w):
    x_min, x_max = x.min(), x.max()
    w_min, w_max = w.min(), w.max()
    
    x_norm = (x - x_min) / (x_max - x_min)
    w_norm = (w - w_min) / (w_max - w_min)
    
    norm_params = {
        'x_min': x_min.item(), 'x_max': x_max.item(),
        'w_min': w_min.item(), 'w_max': w_max.item()
    }
    
    return x_norm, w_norm, norm_params


def denormalize(x_norm, w_norm, norm_params):
    x = x_norm * (norm_params['x_max'] - norm_params['x_min']) + norm_params['x_min']
    w = w_norm * (norm_params['w_max'] - norm_params['w_min']) + norm_params['w_min']
    return x, w

