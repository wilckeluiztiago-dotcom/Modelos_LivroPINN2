"""Utilitários."""
import numpy as np
from typing import Optional

def semente_rng(semente: Optional[int] = None):
    return np.random.default_rng(semente)

def erro_l2(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))
