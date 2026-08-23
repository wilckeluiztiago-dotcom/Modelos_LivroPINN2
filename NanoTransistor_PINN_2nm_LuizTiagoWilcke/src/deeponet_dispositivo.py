"""
Módulo: DeepONet para família de soluções (bias → potencial)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn


class DeepONetDispositivo(nn.Module):
    """
    Branch net: parâmetros de bias (Vgs, Vds)
    Trunk net: coordenadas espaciais (x)
    """
    def __init__(self, branch_dim=2, trunk_dim=1, hidden=64, p=32):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, p)
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, bias, x):
        b = self.branch(bias)          # [B, p]
        t = self.trunk(x)              # [N, p]
        # produto interno
        if bias.dim() == 2 and x.dim() == 2:
            out = torch.einsum("bp,np->bn", b, t) + self.bias
            return out.unsqueeze(-1)
        return (b * t).sum(-1, keepdim=True) + self.bias
