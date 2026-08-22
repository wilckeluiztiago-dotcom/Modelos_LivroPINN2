"""
PINN PyTorch:
  - φ(x)        potencial elétrico (estacionário)
  - σ_H(x, t)   tensão hidrostática (evolução temporal)
"""

import torch
import torch.nn as nn
from typing import List


class RedePotencial(nn.Module):
    """φ(x) — Poisson/Ohm 1D."""

    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [1, 32, 32, 1]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RedeTensao(nn.Module):
    """σ_H(x, t)."""

    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [2, 48, 48, 48, 1]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)
