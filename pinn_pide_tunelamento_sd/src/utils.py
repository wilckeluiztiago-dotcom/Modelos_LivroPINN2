"""Utilitários: amostragem e métricas."""
import numpy as np
from typing import Optional

def amostragem_uniforme(n: int, a: float, b: float, semente: Optional[int] = None) -> np.ndarray:
    g = np.random.default_rng(semente)
    return g.uniform(a, b, size=n)

def amostragem_lhs_1d(n: int, a: float, b: float, semente: Optional[int] = None) -> np.ndarray:
    g = np.random.default_rng(semente)
    u = (np.arange(n) + g.random(n)) / n
    g.shuffle(u)
    return a + u * (b - a)

def erro_l2(u, v) -> float:
    return float(np.sqrt(np.mean((u - v) ** 2)))
