"""
Módulo 17: Calibração de Volatilidade HJM via PINN
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import numpy as np
from .config import CONFIG
from .matematica_hjm import volatilidade_hjm


class ParametroVolatilidade(nn.Module):
    """Parâmetro aprendível de volatilidade (exemplo simples)."""

    def __init__(self, sigma_inicial: float = 0.01):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.tensor(float(np.log(sigma_inicial))))

    def forward(self, t, T):
        sigma = torch.exp(self.log_sigma)
        return sigma * torch.ones_like(t)


# Nota: calibração completa exigiria dados de mercado de swaptions/caps
# e seria feita minimizando o residual da EDP + erro de preços de mercado.
