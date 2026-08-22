# -*- coding: utf-8 -*-
"""
Módulo 12: Função de Fischer-Burmeister para Complementaridade de Coulomb
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import DTYPE

def fischer_burmeister(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    ψ(a,b) = a + b - √(a² + b² + ε)
    Garante a ≥ 0, b ≥ 0, a·b = 0 de forma diferenciável.
    Usado para bloqueio de Coulomb (região de estabilidade).
    """
    return a + b - torch.sqrt(a**2 + b**2 + eps)
