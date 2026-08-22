# -*- coding: utf-8 -*-
"""
Módulo 06: Amostragem Latin Hypercube Sampling (LHS) Multidimensional
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Dict, Tuple
from .constantes_fisicas import DTYPE, DEVICE

def amostragem_lhs(
    n_pontos: int,
    limites: Dict[str, Tuple[float, float]],
    semente: int = 42
) -> Dict[str, torch.Tensor]:
    """
    Gera pontos de colocalização no espaço (V_D, V_G, q, t) via LHS
    livre de malha (meshless).
    """
    torch.manual_seed(semente)
    dim = len(limites)
    amostras = torch.rand(n_pontos, dim, dtype=DTYPE, device=DEVICE)
    for d in range(dim):
        perm = torch.randperm(n_pontos, device=DEVICE)
        amostras[:, d] = (perm.float() + amostras[:, d]) / float(n_pontos)

    resultado = {}
    for i, (nome, (low, high)) in enumerate(limites.items()):
        resultado[nome] = low + (high - low) * amostras[:, i]
    return resultado
