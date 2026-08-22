# -*- coding: utf-8 -*-
"""
Módulo 18: fPINN para Difusão de Carga Sub-Ôhmica (Hurst H < 1/2)
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class fPINNRuido(nn.Module):
    def __init__(self, H: float = 0.3, camadas: list = None):
        super().__init__()
        if camadas is None:
            camadas = [64, 64]
        self.H = H
        self.alpha = 2.0 * H
        self.rede = criar_mlp(2, 1, camadas, "siren")

    def forward(self, t: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.rede(torch.cat([t, q], dim=-1))
