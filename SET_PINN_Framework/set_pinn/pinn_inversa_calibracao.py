# -*- coding: utf-8 -*-
"""
Módulo 27: PINN Inversa para Calibração de Parâmetros do Dispositivo
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class PINNInversaCalibracao(nn.Module):
    """
    Descobre C_G, C_S, C_D, R_T a partir de curvas I-V experimentais ruidosas.
    """
    def __init__(self):
        super().__init__()
        self.parametros = nn.Parameter(
            torch.tensor([5e-17, 1e-16, 1e-16, 2e5], dtype=DTYPE, device=DEVICE)
        )
        self.rede = criar_mlp(2, 1, [64, 64], "tanh")

    def forward(self, V_D: torch.Tensor, V_G: torch.Tensor) -> torch.Tensor:
        return self.rede(torch.cat([V_D, V_G], dim=-1))
