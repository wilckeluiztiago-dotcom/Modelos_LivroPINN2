"""
Módulo: Residual da Equação de Poisson – formulação normalizada rigorosa
Autor: Luiz Tiago Wilcke

Normalização:
  x* = x / L          (L = comprimento do canal)
  φ* = φ / VT         (VT = kT/q)
  n* = n / N_ref      (N_ref = concentração de referência, tipicamente N_D,SD)
  λ² = ε VT / (q N_ref L²)   →  residual = d²φ*/dx*² + (1/λ²)(p*-n*+Nd*-Na*) = 0
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualPoisson:
    def __init__(self, mat: ParametrosSilicio = None, L_nm: float = 14.0,
                 N_ref: float = 2.0e26, T: float = 300.0):
        self.mat = mat or ParametrosSilicio()
        self.L = L_nm * 1e-9
        self.N_ref = N_ref
        self.T = T
        self.VT = self.mat.VT(T)
        self.eps = self.mat.epsilon_si()
        self.q = self.mat.q
        # parâmetro de Debye adimensional
        self.lambda2 = (self.eps * self.VT) / (self.q * self.N_ref * self.L**2)

    def residual(self, phi_star: torch.Tensor, n_star: torch.Tensor,
                 p_star: torch.Tensor, Nd_star: torch.Tensor,
                 Na_star: torch.Tensor, x_star: torch.Tensor) -> torch.Tensor:
        """
        Residual normalizado:
          d²φ*/dx*² + (1/λ²)(p* - n* + Nd* - Na*) = 0
        """
        dphi = torch.autograd.grad(
            phi_star, x_star, grad_outputs=torch.ones_like(phi_star),
            create_graph=True, retain_graph=True
        )[0]
        d2phi = torch.autograd.grad(
            dphi, x_star, grad_outputs=torch.ones_like(dphi),
            create_graph=True, retain_graph=True
        )[0]

        rho_star = p_star - n_star + Nd_star - Na_star
        residual = d2phi + rho_star / self.lambda2
        return residual

    def perda(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.mean(residual**2)

    def desnormalizar_potencial(self, phi_star: torch.Tensor) -> torch.Tensor:
        return phi_star * self.VT

    def desnormalizar_densidade(self, n_star: torch.Tensor) -> torch.Tensor:
        return n_star * self.N_ref
