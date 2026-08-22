import torch
import numpy as np
from typing import Optional

def set_seed(semente: int = 42):
    torch.manual_seed(semente)
    np.random.seed(semente)

def device_padrao():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
