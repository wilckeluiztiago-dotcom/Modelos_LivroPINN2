"""
PINN radial: r → F_j(r) para canais de simetria A1, T2, E.
Energias E_j como parâmetros treináveis.
"""

import torch
import torch.nn as nn
from typing import List, Dict


class RedeEnvelope(nn.Module):
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

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        # envelope que some em r=0 via r * f(r) para regularidade
        return r * self.net(r)


class BancoEMA(nn.Module):
    """Três canais: A1, T2, E."""

    SIMETRIAS = ["A1", "T2", "E"]

    def __init__(self, camadas: List[int] = None):
        super().__init__()
        self.envelopes = nn.ModuleDict({
            s: RedeEnvelope(camadas) for s in self.SIMETRIAS
        })
        # energias (negativas = ligadas); inicializa perto dos alvos normalizados
        self.log_neg_E = nn.ParameterDict({
            "A1": nn.Parameter(torch.tensor(0.0)),   # E = -exp(.)
            "T2": nn.Parameter(torch.tensor(-0.3)),
            "E": nn.Parameter(torch.tensor(-0.4)),
        })

    def energia(self, s: str) -> torch.Tensor:
        return -torch.exp(self.log_neg_E[s])

    def F(self, s: str, r: torch.Tensor) -> torch.Tensor:
        return self.envelopes[s](r)

    def energias_dict(self) -> Dict[str, float]:
        return {s: float(self.energia(s).detach()) for s in self.SIMETRIAS}
