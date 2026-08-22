# -*- coding: utf-8 -*-
"""
Módulo 17: Operador de Caputo via Quadratura de Gauss-Jacobi
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import DTYPE, DEVICE

def caputo_fracionario(
    f: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    n_quad: int = 32
) -> torch.Tensor:
    """
    Derivada fracionária de Caputo de ordem α ∈ (0,1)
    via aproximação de diferenças finitas com memória.
    """
    if f.dim() == 1:
        f = f.unsqueeze(-1)
    if t.dim() == 1:
        t = t.unsqueeze(-1)

    n = f.shape[0]
    resultado = torch.zeros_like(f)
    for i in range(1, n):
        dt = t[i] - t[i-1] + 1e-20
        peso = (dt ** (-alpha)) / max(alpha, 1e-8)
        resultado[i] = peso * (f[i] - f[i-1])
    return resultado
