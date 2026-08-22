# -*- coding: utf-8 -*-
"""
Módulo 16: Solver McKean-Vlasov de Interação Elétron-Elétron
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class PINNMcKeanVlasov(nn.Module):
    def __init__(self, camadas: list = None):
        super().__init__()
        if camadas is None:
            camadas = [64, 64]
        self.rede = criar_mlp(3, 1, camadas, "swish")

    def forward(self, t: torch.Tensor, q: torch.Tensor, campo: torch.Tensor) -> torch.Tensor:
        return self.rede(torch.cat([t, q, campo], dim=-1))
