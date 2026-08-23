"""
Módulo: Bayesian PINN (quantificação de incerteza)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn


class BayesianPINN(nn.Module):
    """
    PINN com dropout de Monte-Carlo para estimativa de incerteza.
    """
    def __init__(self, in_dim=1, out_dim=2, hidden=128, p_drop=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)

    def predicao_incerteza(self, x, n_amostras=30):
        self.train()  # ativa dropout
        preds = torch.stack([self.forward(x) for _ in range(n_amostras)])
        media = preds.mean(0)
        std = preds.std(0)
        return media, std
