import torch
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training: # Dont dropout path while in inference mode
        return x
    keep_prob = 1 - drop_prob # Complment of drop rate
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device) # Genreate random tensor of vals between 0 and 1
    random_tensor.floor_()  # binarize (all values over 1 will become 1 and all less will be 0)
    output = x.div(keep_prob) * random_tensor # Mask out droped paths
    return output

class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample.

    Params:
        drop_prob: Probabilty of dropping a given path. If None, all paths will remain. Default is None
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
