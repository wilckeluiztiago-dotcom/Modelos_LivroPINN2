"""
Módulo: Mobilidade de campo alto – Caughey-Thomas + Lombardi (Si)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class MobilidadeCampoAlto:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.mu0 = self.mat.mobilidade_n0
        self.vsat = self.mat.vsat_n
        self.beta = 2.0

    def caughey_thomas(self, E: torch.Tensor) -> torch.Tensor:
        """μ(E) = μ0 / [1 + (μ0|E|/vsat)^β]^(1/β)"""
        E_abs = torch.abs(E) + 1e-6
        razao = (self.mu0 * E_abs / self.vsat)**self.beta
        return self.mu0 / (1.0 + razao)**(1.0 / self.beta)

    def lombardi(self, E_perp: torch.Tensor, T: float = 300.0,
                 N_a: float = 1e21) -> torch.Tensor:
        """
        Modelo Lombardi (superfície + fônons + impurezas).
        Retorna mobilidade efetiva (m²/V·s).
        """
        # scattering de superfície
        A = 4.75e6   # cm/s → convertido para m
        B = 1.0e5
        E_p = torch.abs(E_perp) + 1e3
        mu_sr = (A + B * (N_a / 1e23)**0.125) / E_p * 1e-2  # ajuste de unidade

        # fônons
        mu_ph = self.mu0 * (T / 300.0)**(-2.15)

        # impurezas (Caughey-Thomas simplificado)
        mu_imp = self.mu0 * 0.5

        return 1.0 / (1.0 / (mu_sr + 1e-8) + 1.0 / mu_ph + 1.0 / mu_imp)
