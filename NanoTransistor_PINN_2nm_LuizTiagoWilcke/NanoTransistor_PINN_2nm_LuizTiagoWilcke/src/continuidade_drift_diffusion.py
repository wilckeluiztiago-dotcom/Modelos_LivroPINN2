"""
Módulo: Equações de Continuidade e Drift-Diffusion com correção quântica
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualContinuidade:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.q = self.mat.q
        self.mu_n = self.mat.mobilidade_n0
        self.D_n = self.mu_n * self.mat.VT()  # Einstein

    def corrente_eletrons(self, n, phi, x):
        """J_n = q μ_n n E + q D_n ∇n  (E = -∇φ)"""
        dn_dx = torch.autograd.grad(n, x, grad_outputs=torch.ones_like(n),
                                    create_graph=True, retain_graph=True)[0]
        dphi_dx = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                      create_graph=True, retain_graph=True)[0]
        E = -dphi_dx
        J_n = self.q * self.mu_n * n * E + self.q * self.D_n * dn_dx
        return J_n

    def residual_continuidade(self, n, phi, x, G=0.0, R=0.0):
        """∂n/∂t ≈ 0 em regime estacionário → (1/q) ∇·J_n + G - R = 0"""
        J_n = self.corrente_eletrons(n, phi, x)
        dJn_dx = torch.autograd.grad(J_n, x, grad_outputs=torch.ones_like(J_n),
                                     create_graph=True, retain_graph=True)[0]
        residual = (1.0 / self.q) * dJn_dx + G - R
        return residual

    def perda(self, residual):
        return torch.mean(residual**2)
