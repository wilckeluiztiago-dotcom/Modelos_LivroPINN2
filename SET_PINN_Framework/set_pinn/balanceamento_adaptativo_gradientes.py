# -*- coding: utf-8 -*-
"""
Módulo 21: Balanceamento Auto-Adaptativo de Pesos de Perda
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Dict

def atualizar_lambdas(
    perdas: Dict[str, torch.Tensor],
    grads: Dict[str, torch.Tensor],
    eps: float = 1e-8
) -> Dict[str, float]:
    """
    λ_i ∝ 1 / ||∇_θ L_i||  (normalização por norma de gradiente)
    """
    normas = {k: torch.norm(g).item() + eps for k, g in grads.items()}
    total = sum(normas.values())
    n = len(normas)
    return {k: total / (v * n) for k, v in normas.items()}
