"""
Utilitários: amostragem LHS, métricas e grades.
Base: Caps. 2 e 3 do livro de PINNs (Luiz Tiago Wilcke).
"""

import numpy as np
from typing import Tuple, Optional


def amostragem_lhs(n: int, limites: np.ndarray, semente: Optional[int] = None) -> np.ndarray:
    """
    Latin Hypercube Sampling (Cap. 3.5 do livro).
    limites: array (d, 2) com [min, max] por dimensão.
    Retorna: (n, d)
    """
    gerador = np.random.default_rng(semente)
    d = limites.shape[0]
    pontos = np.zeros((n, d))
    for j in range(d):
        intervalos = np.linspace(0, 1, n + 1)
        u = gerador.uniform(intervalos[:-1], intervalos[1:])
        gerador.shuffle(u)
        pontos[:, j] = limites[j, 0] + u * (limites[j, 1] - limites[j, 0])
    return pontos


def amostragem_uniforme(n: int, limites: np.ndarray, semente: Optional[int] = None) -> np.ndarray:
    gerador = np.random.default_rng(semente)
    d = limites.shape[0]
    return gerador.uniform(limites[:, 0], limites[:, 1], size=(n, d))


def erro_l2(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.sqrt(np.mean((u - v) ** 2)))


def erro_max(u: np.ndarray, v: np.ndarray) -> float:
    return float(np.max(np.abs(u - v)))


def normalizar(x: np.ndarray, limites: np.ndarray) -> np.ndarray:
    """Mapeia x para [-1, 1] por dimensão."""
    mid = 0.5 * (limites[:, 0] + limites[:, 1])
    half = 0.5 * (limites[:, 1] - limites[:, 0])
    return (x - mid) / (half + 1e-12)
