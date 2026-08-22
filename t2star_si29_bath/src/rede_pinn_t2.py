"""
PINN: t → (Re ρ_↑↓, Im ρ_↑↓)  coerência do spin ³¹P
ou diretamente ⟨S_x(t)⟩.
"""

import torch
import torch.nn as nn
from typing import List


class RedePINN_T2(nn.Module):
    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [1, 48, 48, 1]
        layers = []
        for i in range(len(camadas) - 1):
            layers.append(nn.Linear(camadas[i], camadas[i + 1]))
            if i < len(camadas) - 2:
                layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)
        # parâmetro treinável log T2*
        self.log_T2s = nn.Parameter(torch.tensor(0.0))
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def T2_star(self) -> torch.Tensor:
        return torch.exp(self.log_T2s)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """⟨S_x(t)⟩ predito (positivo via softplus map opcional)."""
        raw = self.net(t)
        # coerência entre -1 e 1
        return torch.tanh(raw)
