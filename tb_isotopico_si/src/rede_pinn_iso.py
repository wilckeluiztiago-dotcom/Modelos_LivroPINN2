"""
PINN: índice de sítio / coordenada → amplitudes C_i do estado fundamental
(ou estado j), com E treinável.
"""

import torch
import torch.nn as nn
from typing import List


class RedePINN_Iso(nn.Module):
    def __init__(self, n_sites: int = 21, camadas: List[int] = None):
        super().__init__()
        self.n = n_sites
        if camadas is None:
            camadas = [1, 48, 48, 1]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        self.log_neg_E = nn.Parameter(torch.tensor(0.5))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def energia(self) -> torch.Tensor:
        return -torch.exp(self.log_neg_E)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (n,1) posições normalizadas → C (n,1)"""
        return self.net(x)

    def vetor_C(self, device=None) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        x = torch.linspace(0, 1, self.n, device=device).reshape(-1, 1)
        C = self.forward(x).squeeze()
        return C
