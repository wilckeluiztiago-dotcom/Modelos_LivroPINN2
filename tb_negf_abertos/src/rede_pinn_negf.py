"""
PINN: E → G^R(E) como matriz complexa n×n (parte real e imaginária).
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_NEGF(nn.Module):
    def __init__(self, n: int = 7, camadas: List[int] = None):
        super().__init__()
        self.n = n
        out = 2 * n * n  # Re e Im
        if camadas is None:
            camadas = [1, 64, 64, 64, out]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, E: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        E: (B,1) → ReG, ImG cada (B, n, n)
        """
        out = self.net(E)
        B = E.shape[0]
        n = self.n
        Re = out[:, : n * n].reshape(B, n, n)
        Im = out[:, n * n :].reshape(B, n, n)
        return Re, Im

    def G_complex(self, E: torch.Tensor) -> torch.Tensor:
        Re, Im = self.forward(E)
        return torch.complex(Re, Im)
