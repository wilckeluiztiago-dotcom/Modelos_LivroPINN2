# -*- coding: utf-8 -*-
"""
Módulo 15: DeepONet para Mapeamento Contínuo de Condutância
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class DeepONetTransporte(nn.Module):
    """
    Branch: processa perfil V_G(t) / dopagem
    Trunk: processa coordenadas (x,t)
    Saída: G(V_D, V_G) ou I(V_D, V_G)
    """
    def __init__(self, dim_branch: int = 64, dim_trunk: int = 2, dim_latente: int = 64):
        super().__init__()
        self.branch = criar_mlp(dim_branch, dim_latente, [128, 128], "tanh")
        self.trunk = criar_mlp(dim_trunk, dim_latente, [128, 128], "tanh")
        self.bias = nn.Parameter(torch.zeros(1, dtype=DTYPE, device=DEVICE))

    def forward(self, u_branch: torch.Tensor, y_trunk: torch.Tensor) -> torch.Tensor:
        b = self.branch(u_branch)
        t = self.trunk(y_trunk)
        return torch.sum(b * t, dim=-1, keepdim=True) + self.bias
