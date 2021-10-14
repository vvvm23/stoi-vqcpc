import torch
import numpy as np
from pathlib import Path

def get_parameter_count(net: torch.nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)
def get_device(cpu):
    if cpu or not torch.cuda.is_available(): return torch.device('cpu')
    return torch.device('cuda')

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
