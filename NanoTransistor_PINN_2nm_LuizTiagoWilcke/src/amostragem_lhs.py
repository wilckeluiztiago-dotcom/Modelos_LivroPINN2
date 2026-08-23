"""
Módulo: Amostragem Latin Hypercube Sampling (LHS)
Autor: Luiz Tiago Wilcke
"""

import torch
import numpy as np


def amostragem_lhs(n_pontos: int, dim: int = 1, seed: int = None) -> torch.Tensor:
    """Latin Hypercube Sampling no hipercubo [0,1]^dim."""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        amostra = sampler.random(n=n_pontos)
        return torch.tensor(amostra, dtype=torch.float32)
    except Exception:
        # fallback: amostragem estratificada simples
        if seed is not None:
            torch.manual_seed(seed)
        amostra = torch.rand(n_pontos, dim)
        for d in range(dim):
            amostra[:, d] = (torch.argsort(amostra[:, d]).float() + torch.rand(n_pontos)) / n_pontos
        return amostra


def amostragem_fronteira(n_pontos: int, valor: float = 0.0) -> torch.Tensor:
    return torch.full((n_pontos, 1), valor, dtype=torch.float32)
