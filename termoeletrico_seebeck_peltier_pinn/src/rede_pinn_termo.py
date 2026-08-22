"""
PINN PyTorch: (x) → (φ(x), T(x))  [regime estacionário 1D]
"""

import torch
import torch.nn as nn
from typing import List


class RedePINN_Termo(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [1, 64, 64, 64, 2]  # x → (φ, T)
        camadas_nn = []
        for i in range(len(camadas) - 1):
            camadas_nn.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                camadas_nn.append(nn.Tanh())
        self.net = nn.Sequential(*camadas_nn)
        self._init_pesos()

    def _init_pesos(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N,1)
        retorna (N,2) = (φ, T) com T > 0 via softplus no canal T.
        """
        out = self.net(x)
        phi = out[:, 0:1]
        T = torch.nn.functional.softplus(out[:, 1:2]) + 0.1
        return torch.cat([phi, T], dim=1)

    def campos(self, x: torch.Tensor):
        out = self.forward(x)
        return out[:, 0:1], out[:, 1:2]
