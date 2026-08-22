# -*- coding: utf-8 -*-
"""
Módulo 22: Reamostragem Adaptativa Focada em Diamantes de Coulomb
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import DTYPE, DEVICE

def reamostrar_residuos(
    pontos: torch.Tensor,
    residuos: torch.Tensor,
    n_novos: int,
    temperatura: float = 1.0
) -> torch.Tensor:
    """
    Amostra novos pontos com probabilidade ∝ |resíduo|^temperatura
    (enriquecimento nas bordas dos diamantes).
    """
    pesos = torch.abs(residuos).flatten() ** temperatura
    pesos = pesos / (pesos.sum() + 1e-12)
    indices = torch.multinomial(pesos, n_novos, replacement=True)
    return pontos[indices]
