"""
PINN: x → (φ, n, p)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_BTBT(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [1, 64, 64, 64, 3]
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        phi = out[:, 0:1]
        n = torch.nn.functional.softplus(out[:, 1:2]) + 1e-3
        p = torch.nn.functional.softplus(out[:, 2:3]) + 1e-3
        return torch.cat([phi, n, p], dim=1)

    def campos(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        o = self.forward(x)
        return o[:, 0:1], o[:, 1:2], o[:, 2:3]
