# -*- coding: utf-8 -*-
"""
Módulo 20: PINN Multi-Head para Estados de Carga N = 0,1,2,...
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class PINNMultiCabecas(nn.Module):
    def __init__(self, n_estados: int = 5, dim_tronco: int = 64):
        super().__init__()
        self.tronco = criar_mlp(3, dim_tronco, [128, 128], "tanh")
        self.cabecas = nn.ModuleList([
            nn.Linear(dim_tronco, 1, dtype=DTYPE, device=DEVICE) for _ in range(n_estados)
        ])

    def forward(self, t: torch.Tensor, V_D: torch.Tensor, V_G: torch.Tensor) -> torch.Tensor:
        h = self.tronco(torch.cat([t, V_D, V_G], dim=-1))
        saidas = [cabeca(h) for cabeca in self.cabecas]
        return torch.cat(saidas, dim=-1)
