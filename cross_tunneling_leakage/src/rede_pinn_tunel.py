"""
PINN: (z, t) → (V1, V2, I1, I2)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_Tunel(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [2, 64, 64, 64, 4]  # (z,t) → (V1,V2,I1,I2)
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

    def forward(self, zt: torch.Tensor) -> torch.Tensor:
        return self.net(zt)

    def campos(self, zt: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        out = self.forward(zt)
        return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]
