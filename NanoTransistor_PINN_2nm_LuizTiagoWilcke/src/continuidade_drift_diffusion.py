"""
Módulo: Continuidade e Drift-Diffusion – formulação normalizada
Autor: Luiz Tiago Wilcke

Corrente de elétrons (normalizada):
  J_n* = μ_n* (n* E* + ∇n*)     (Einstein: D = μ VT)
  E* = -∇φ*
Em regime estacionário: ∇·J_n* = 0  (sem geração/recombinação líquida)
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualContinuidade:
    def __init__(self, mat: ParametrosSilicio = None, L_nm: float = 14.0,
                 N_ref: float = 2.0e26, T: float = 300.0):
        self.mat = mat or ParametrosSilicio()
        self.L = L_nm * 1e-9
        self.N_ref = N_ref
        self.T = T
        self.VT = self.mat.VT(T)
        self.mu_n = self.mat.mobilidade_n0
        # mobilidade normalizada (adimensional no residual)
        self.mu_star = 1.0  # já absorvida na escala de corrente

    def corrente_eletrons(self, n_star: torch.Tensor, phi_star: torch.Tensor,
                          x_star: torch.Tensor) -> torch.Tensor:
        """J_n* = n* (-dφ*/dx*) + dn*/dx*   (forma Slotboom / Einstein)"""
        dn = torch.autograd.grad(
            n_star, x_star, grad_outputs=torch.ones_like(n_star),
            create_graph=True, retain_graph=True
        )[0]
        dphi = torch.autograd.grad(
            phi_star, x_star, grad_outputs=torch.ones_like(phi_star),
            create_graph=True, retain_graph=True
        )[0]
        E_star = -dphi
        J_n_star = n_star * E_star + dn
        return J_n_star

    def residual_continuidade(self, n_star: torch.Tensor, phi_star: torch.Tensor,
                              x_star: torch.Tensor,
                              G_star: float = 0.0, R_star: float = 0.0) -> torch.Tensor:
        """∇·J_n* - (G*-R*) = 0  (estacionário)"""
        J = self.corrente_eletrons(n_star, phi_star, x_star)
        dJ = torch.autograd.grad(
            J, x_star, grad_outputs=torch.ones_like(J),
            create_graph=True, retain_graph=True
        )[0]
        return dJ - (G_star - R_star)

    def perda(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.mean(residual**2)

    def corrente_fisica_A_por_m(self, J_star: torch.Tensor) -> torch.Tensor:
        """Converte corrente normalizada para A/m (por largura)."""
        # J = q * N_ref * μ * VT / L * J*
        escala = (self.mat.q * self.N_ref * self.mu_n * self.VT) / self.L
        return J_star * escala
