# -*- coding: utf-8 -*-
"""
Módulo 07: Arquiteturas Densas e Ativações C∞
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from typing import List
from .constantes_fisicas import DTYPE, DEVICE

class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

class SIREN(nn.Module):
    def __init__(self, omega0: float = 30.0):
        super().__init__()
        self.omega0 = omega0
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega0 * x)

def criar_mlp(
    dim_entrada: int,
    dim_saida: int,
    camadas_ocultas: List[int],
    ativacao: str = "tanh",
    omega0: float = 30.0
) -> nn.Module:
    """MLP totalmente conectada com ativação infinitamente diferenciável."""
    camadas = []
    dims = [dim_entrada] + camadas_ocultas + [dim_saida]
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i+1], dtype=DTYPE, device=DEVICE)
        if ativacao == "siren" and i == 0:
            nn.init.uniform_(linear.weight, -1.0 / dims[i], 1.0 / dims[i])
        else:
            nn.init.xavier_normal_(linear.weight)
        camadas.append(linear)
        if i < len(dims) - 2:
            if ativacao == "tanh":
                camadas.append(nn.Tanh())
            elif ativacao == "swish":
                camadas.append(Swish())
            elif ativacao == "siren":
                camadas.append(SIREN(omega0))
    return nn.Sequential(*camadas).to(DEVICE)
