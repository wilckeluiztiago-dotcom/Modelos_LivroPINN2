"""
PINN: r=(x,y,z) → amplitudes orbitais contínuas C_α(r) (10 canais).
Usada para interpolar / regularizar o estado fundamental TB.
"""

import torch
import torch.nn as nn
from typing import List


class RedePINN_TB(nn.Module):
    def __init__(self, n_orb: int = 10, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [3, 64, 64, n_orb]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_neg_E = nn.Parameter(torch.tensor(0.5))  # E = -exp
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def energia(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_E)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        """(N,3) → (N, n_orb)"""
        return self.net(r)
