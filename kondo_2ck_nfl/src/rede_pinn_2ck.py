"""
PINN:
  1) G_θ(V, T)  — condutância diferencial
  2) ρ_θ(t)     — matriz densidade 2×2 do spin (parametrizada)
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class RedePINN_G(nn.Module):
    """(V, T) → G"""

    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [2, 48, 48, 1]
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

    def forward(self, VT: torch.Tensor) -> torch.Tensor:
        # G ≥ 0
        return torch.nn.functional.softplus(self.net(VT))


class RedePINN_Rho(nn.Module):
    """
    t → parâmetros de ρ 2×2 hermitiana positiva:
      ρ = [[a, c+id], [c-id, 1-a]] com a∈(0,1)
    """

    def __init__(self, camadas: List[int] = None):
        super().__init__()
        if camadas is None:
            camadas = [1, 48, 48, 3]
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

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        raw = self.net(t)
        a = torch.sigmoid(raw[:, 0:1])
        c = 0.2 * torch.tanh(raw[:, 1:2])
        d = 0.2 * torch.tanh(raw[:, 2:3])
        return torch.cat([a, c, d], dim=1)

    def matriz_rho(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retorna elementos ρ00, ρ01_re, ρ01_im, ρ11"""
        p = self.forward(t)
        a, c, d = p[:, 0:1], p[:, 1:2], p[:, 2:3]
        return a, c, d, 1.0 - a
