"""
Módulo: Residual da Equação de Schrödinger (efetiva-massa)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualSchrodinger:
    """
    Residual 1D da equação de Schrödinger efetiva-massa na direção de confinamento.
    """
    def __init__(self, mat: ParametrosSilicio = None, t_nm: float = 2.0):
        self.mat = mat or ParametrosSilicio()
        self.hbar = self.mat.hbar
        self.m_star = self.mat.massa_efetiva_kg()
        self.q = self.mat.q
        self.t = t_nm * 1e-9  # metros

    def residual(self, psi: torch.Tensor, E: torch.Tensor, V: torch.Tensor,
                 y: torch.Tensor) -> torch.Tensor:
        """
        - (ħ²/2m*) d²ψ/dy² + V ψ - E ψ = 0
        y normalizado [0,1]; E e V em Joules ou eV consistentes.
        """
        dpsi_dy = torch.autograd.grad(
            psi, y, grad_outputs=torch.ones_like(psi),
            create_graph=True, retain_graph=True
        )[0]
        d2psi_dy2 = torch.autograd.grad(
            dpsi_dy, y, grad_outputs=torch.ones_like(dpsi_dy),
            create_graph=True, retain_graph=True
        )[0]

        # escala espacial
        escala = (self.hbar**2) / (2.0 * self.m_star * self.t**2)
        residual = -escala * d2psi_dy2 + (V - E) * psi
        return residual

    def perda(self, residual: torch.Tensor, psi: torch.Tensor = None,
              normalizar: bool = True) -> torch.Tensor:
        loss = torch.mean(residual**2)
        if normalizar and psi is not None:
            # penalidade ∫|ψ|² dy ≈ 1
            norma = torch.mean(psi**2) - 1.0
            loss = loss + 5.0 * norma**2
        return loss


if __name__ == "__main__":
    print("ResidualSchrodinger carregado - Luiz Tiago Wilcke")
