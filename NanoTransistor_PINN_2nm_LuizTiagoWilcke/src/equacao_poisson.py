"""
Módulo 04: Residual da Equação de Poisson
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio

class ResidualPoisson:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.eps = self.mat.epsilon_si()
        self.q = self.mat.q

    def residual(self, phi, n, p, Nd, Na, x):
        """
        ∇·(ε ∇φ) + ρ = 0
        Em 1D: ε d²φ/dx² + q (p - n + Nd - Na) = 0
        Retorna residual (deve → 0).
        """
        # diferenciação automática
        dphi_dx = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                      create_graph=True, retain_graph=True)[0]
        d2phi_dx2 = torch.autograd.grad(dphi_dx, x, grad_outputs=torch.ones_like(dphi_dx),
                                        create_graph=True, retain_graph=True)[0]
        rho = self.q * (p - n + Nd - Na)
        residual = self.eps * d2phi_dx2 + rho
        return residual

    def perda(self, residual):
        return torch.mean(residual**2)
