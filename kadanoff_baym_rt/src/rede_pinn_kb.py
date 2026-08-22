"""
PINN: (t1, t2) → (Re G^<, Im G^<, Re G^R, Im G^R)
para um orbital (G escalar complexo).
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_KB(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [2, 64, 64, 64, 4]
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

    def forward(self, t12: torch.Tensor) -> torch.Tensor:
        return self.net(t12)

    def G_lesser(self, t12: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        o = self.forward(t12)
        return o[:, 0:1], o[:, 1:2]

    def G_retarded(self, t12: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        o = self.forward(t12)
        # G^R causal: preferencialmente ~0 para t1 < t2
        return o[:, 2:3], o[:, 3:4]
