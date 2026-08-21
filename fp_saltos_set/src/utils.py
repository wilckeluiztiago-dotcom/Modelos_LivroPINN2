import numpy as np
from typing import Optional

def rng(semente: Optional[int] = None):
    return np.random.default_rng(semente)
