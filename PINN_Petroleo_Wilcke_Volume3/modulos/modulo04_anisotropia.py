# =============================================================================
# Módulo 04: Escoamento em Meios Porosos Anisotrópicos e Inversão de Permeabilidade
# Autor: Luiz Tiago Wilcke
# Capítulo 4
# =============================================================================
"""Tensor de permeabilidade, PINN inversa, regularização TV, B-PINN."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
from ..config.configuracoes import FISICA, PINN
from ..utils.utilitarios import gradiente_autograd, LOGGER
from .modulo01_fundamentos import RedeBasePINN, FundamentosReservatorio

class AnisotropiaPermeabilidade:
    def __init__(self):
        self.fund = FundamentosReservatorio()
        LOGGER.info("AnisotropiaPermeabilidade - Luiz Tiago Wilcke")

    def tensor_k(self, kx, ky, kz):
        return torch.diag(torch.tensor([kx, ky, kz], dtype=torch.float32))

    def residuo_darcy_aniso(self, p, x, y, z, kx, ky, kz, phi, ct, mu):
        p_x = gradiente_autograd(p, x)
        p_y = gradiente_autograd(p, y)
        p_z = gradiente_autograd(p, z)
        p_t = gradiente_autograd(p, z)  # placeholder t
        div = (kx/mu)*gradiente_autograd(p_x, x) + (ky/mu)*gradiente_autograd(p_y, y) + (kz/mu)*gradiente_autograd(p_z, z)
        return div - phi*ct*p_t

class PINNInversaPermeabilidade(RedeBasePINN):
    def __init__(self):
        super().__init__(dim_entrada=3, dim_saida=1)
        self.log_kx = nn.Parameter(torch.tensor(0.0))
        self.log_ky = nn.Parameter(torch.tensor(0.0))
        self.log_kz = nn.Parameter(torch.tensor(-2.0))

    def permeabilidades(self):
        return torch.exp(self.log_kx), torch.exp(self.log_ky), torch.exp(self.log_kz)

    def perda_conjunta(self, pred, alvo, residuo, lambda_tv=0.01):
        l_data = torch.mean((pred - alvo)**2)
        l_phys = torch.mean(residuo**2)
        # Regularização TV simplificada
        l_tv = lambda_tv * (torch.abs(self.log_kx) + torch.abs(self.log_ky) + torch.abs(self.log_kz))
        return l_data + l_phys + l_tv
