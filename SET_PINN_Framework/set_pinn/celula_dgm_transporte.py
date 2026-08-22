# -*- coding: utf-8 -*-
"""
Módulo 08: Célula Deep Galerkin Method (DGM) com Portas Multiplicativas
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .constantes_fisicas import DTYPE, DEVICE

class CelulaDGM(nn.Module):
    """
    Bloco DGM: Z, G, R, H com portas multiplicativas para estabilizar
    derivadas de 2ª ordem na equação mestra / Fokker-Planck.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.Z = nn.Linear(2 * dim, dim, dtype=DTYPE, device=DEVICE)
        self.G = nn.Linear(2 * dim, dim, dtype=DTYPE, device=DEVICE)
        self.R = nn.Linear(2 * dim, dim, dtype=DTYPE, device=DEVICE)
        self.H = nn.Linear(2 * dim, dim, dtype=DTYPE, device=DEVICE)
        self.ativacao = nn.Tanh()

    def forward(self, S: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        entrada = torch.cat([S, X], dim=-1)
        Z = self.ativacao(self.Z(entrada))
        G = self.ativacao(self.G(entrada))
        R = self.ativacao(self.R(entrada))
        H = self.ativacao(self.H(entrada))
        return (1.0 - G) * H + Z * S
