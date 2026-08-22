# -*- coding: utf-8 -*-
"""
Módulo 13: PINN Acoplada Schrödinger-Poisson 1D
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .constantes_fisicas import hbar, m_0, e, epsilon_0, DTYPE, DEVICE

class SolverSchrodingerPoisson(nn.Module):
    def __init__(self, camadas: list = None):
        super().__init__()
        if camadas is None:
            camadas = [64, 64, 64]
        self.rede_psi = criar_mlp(1, 1, camadas, ativacao="tanh")
        self.rede_phi = criar_mlp(1, 1, camadas, ativacao="tanh")

    def forward(self, x: torch.Tensor) -> tuple:
        psi = self.rede_psi(x)
        phi = self.rede_phi(x)
        return psi, phi

    def residuo_acoplado(self, x: torch.Tensor, U_conf: torch.Tensor, N_D: torch.Tensor) -> torch.Tensor:
        psi, phi = self.forward(x)
        dpsi = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi), create_graph=True)[0]
        d2psi = torch.autograd.grad(dpsi, x, grad_outputs=torch.ones_like(dpsi), create_graph=True)[0]
        Hpsi = - (hbar**2 / (2 * m_0)) * d2psi + (-e * phi + U_conf) * psi
        dphi = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
        d2phi = torch.autograd.grad(dphi, x, grad_outputs=torch.ones_like(dphi), create_graph=True)[0]
        residuo_poisson = epsilon_0 * d2phi + e * (psi**2 - N_D)
        return Hpsi + residuo_poisson
