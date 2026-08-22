# -*- coding: utf-8 -*-
"""
Módulo 28: B-PINN / Monte Carlo Dropout para Intervalos de Confiança
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import DTYPE, DEVICE

class BPINN(nn.Module):
    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.rede = nn.Sequential(
            criar_mlp(4, 64, [], "tanh"),
            nn.Dropout(p_drop),
            criar_mlp(64, 64, [], "tanh"),
            nn.Dropout(p_drop),
            nn.Linear(64, 1, dtype=DTYPE, device=DEVICE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.rede(x)

    def predicao_incerteza(self, x: torch.Tensor, n_samples: int = 30) -> tuple:
        self.train()  # mantém dropout ativo
        preds = torch.stack([self.forward(x) for _ in range(n_samples)])
        return preds.mean(0), preds.std(0)
