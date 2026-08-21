"""
Utilitários: amostragem no espaço de fases, métricas.
"""

import numpy as np
from typing import Optional, Tuple


def amostragem_lhs(n: int, limites: np.ndarray, semente: Optional[int] = None) -> np.ndarray:
    """Latin Hypercube Sampling. limites: (d, 2)."""
    gerador = np.random.default_rng(semente)
    d = limites.shape[0]
    pts = np.zeros((n, d))
    for j in range(d):
        u = (np.arange(n) + gerador.random(n)) / n
        gerador.shuffle(u)
        pts[:, j] = limites[j, 0] + u * (limites[j, 1] - limites[j, 0])
    return pts


def amostragem_uniforme(n: int, limites: np.ndarray, semente: Optional[int] = None) -> np.ndarray:
    gerador = np.random.default_rng(semente)
    return gerador.uniform(limites[:, 0], limites[:, 1], size=(n, limites.shape[0]))


def erro_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def normalizar_fase(X: np.ndarray, limites: np.ndarray) -> np.ndarray:
    mid = 0.5 * (limites[:, 0] + limites[:, 1])
    half = 0.5 * (limites[:, 1] - limites[:, 0])
    return (X - mid) / (half + 1e-12)
