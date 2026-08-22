# -*- coding: utf-8 -*-
"""
Módulo 19: Penalização de Conservação de Probabilidade / Carga
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import DTYPE

def perda_conservacao(p: torch.Tensor, dq: float = 1.0) -> torch.Tensor:
    """
    ∫ p(q) dq = 1  →  perda = (Σ p·dq - 1)²
    """
    integral = torch.sum(p) * dq
    return (integral - 1.0) ** 2
