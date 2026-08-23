"""
Módulo: Residual da Equação de Schrödinger (efetiva-massa, normalizado)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualSchrodinger:
    def __init__(self, mat: ParametrosSilicio = None, t_nm: float = 2.0, T: float = 300.0):
        self.mat = mat or ParametrosSilicio()
        self.t = t_nm * 1e-9
        self.hbar = self.mat.hbar
        self.m_star = self.mat.massa_efetiva_kg()
        self.q = self.mat.q
        self.VT = self.mat.VT(T)
        # energia de confinamento característica E0 = ħ²π²/(2m*t²)
        self.E0 = (self.hbar**2 * (3.14159265)**2) / (2 * self.m_star * self.t**2)
        self.E0_eV = self.E0 / self.q

    def residual(self, psi: torch.Tensor, E_star: torch.Tensor, V_star: torch.Tensor,
                 y_star: torch.Tensor) -> torch.Tensor:
        """
        Forma normalizada:
          - (E0 / VT) d²ψ/dy*² + (V* - E*) ψ = 0
        y* ∈ [0,1], V* e E* em unidades de VT.
        """
        dpsi = torch.autograd.grad(psi, y_star, grad_outputs=torch.ones_like(psi),
                                   create_graph=True, retain_graph=True)[0]
        d2psi = torch.autograd.grad(dpsi, y_star, grad_outputs=torch.ones_like(dpsi),
                                    create_graph=True, retain_graph=True)[0]
        coef = self.E0 / (self.q * self.VT)
        residual = -coef * d2psi + (V_star - E_star) * psi
        return residual

    def perda(self, residual: torch.Tensor, psi: torch.Tensor = None) -> torch.Tensor:
        loss = torch.mean(residual**2)
        if psi is not None:
            # normalização ∫|ψ|² dy* = 1
            loss = loss + 10.0 * (torch.mean(psi**2) - 1.0)**2
        return loss
