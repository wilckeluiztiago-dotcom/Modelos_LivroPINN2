"""
Módulo: HJB para Otimização de Layout / Dopagem
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn


class HJBLayout(nn.Module):
    """
    Equação de Hamilton-Jacobi-Bellman para otimização
    de perfil de dopagem (controle ótimo).
    """
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def valor(self, estado):
        return self.net(estado)

    def controle_otimo(self, estado, grad_V):
        """u* = argmin { L + ∇V · f }  (simplificado)."""
        return -torch.tanh(grad_V)  # controle limitado
