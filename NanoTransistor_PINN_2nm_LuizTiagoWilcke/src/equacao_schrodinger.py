"""
Módulo 05: Residual da Equação de Schrödinger (efetiva-massa)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio

class ResidualSchrodinger:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.hbar = self.mat.hbar
        self.m_star = self.mat.m_star_transversal * self.mat.m0
        self.q = self.mat.q

    def residual(self, psi, E, V, y):
        """
        - (ħ²/2m*) d²ψ/dy² + V ψ - E ψ = 0
        y é a direção de confinamento (espessura 2 nm).
        """
        dpsi_dy = torch.autograd.grad(psi, y, grad_outputs=torch.ones_like(psi),
                                      create_graph=True, retain_graph=True)[0]
        d2psi_dy2 = torch.autograd.grad(dpsi_dy, y, grad_outputs=torch.ones_like(dpsi_dy),
                                        create_graph=True, retain_graph=True)[0]
        residual = - (self.hbar**2 / (2 * self.m_star)) * d2psi_dy2 + (V - E) * psi
        return residual

    def perda(self, residual, normalizacao=True, psi=None):
        loss = torch.mean(residual**2)
        if normalizacao and psi is not None:
            # penalidade de normalização ∫|ψ|² = 1
            norm = torch.mean(psi**2) - 1.0
            loss = loss + 10.0 * norm**2
        return loss
