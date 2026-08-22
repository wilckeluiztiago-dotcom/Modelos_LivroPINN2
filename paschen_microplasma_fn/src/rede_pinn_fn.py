"""
PINN: (x, t) → (φ, n_e)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_FN(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [2, 64, 64, 64, 2]
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

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        out = self.net(xt)
        phi = out[:, 0:1]
        ne = torch.nn.functional.softplus(out[:, 1:2]) + 1e-4
        return torch.cat([phi, ne], dim=1)

    def campos(self, xt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.forward(xt)
        return out[:, 0:1], out[:, 1:2]
