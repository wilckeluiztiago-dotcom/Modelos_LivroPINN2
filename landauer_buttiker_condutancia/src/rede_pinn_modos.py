"""
PINN para modos transversais ψ_n(y) e energias E_n.
"""

import torch
import torch.nn as nn
from typing import List


class RedeModo(nn.Module):
    """Uma rede por modo: y → ψ_n(y)."""

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
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.net(y)


class BancoModos(nn.Module):
    """N modos + parâmetros de energia E_n."""

    def __init__(self, n_modos: int = 3, camadas: List[int] = None):
        super().__init__()
        self.n_modos = n_modos
        self.modos = nn.ModuleList([RedeModo(camadas) for _ in range(n_modos)])
        # energias como parâmetros positivos
        self.log_E = nn.Parameter(torch.log(torch.arange(1, n_modos + 1, dtype=torch.float32) ** 2 * 0.5))

    def energias(self) -> torch.Tensor:
        return torch.exp(self.log_E)

    def psi(self, n: int, y: torch.Tensor) -> torch.Tensor:
        return self.modos[n](y)
