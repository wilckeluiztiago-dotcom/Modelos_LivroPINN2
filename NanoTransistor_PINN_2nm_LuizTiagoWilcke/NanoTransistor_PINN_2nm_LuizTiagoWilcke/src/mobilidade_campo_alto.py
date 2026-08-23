"""
Módulo: Mobilidade de Campo Alto (Caughey-Thomas / Lombardi)
Autor: Luiz Tiago Wilcke
"""

import torch


class MobilidadeCampoAlto:
    def __init__(self, mu0: float = 0.14, vsat: float = 1e5, beta: float = 2.0):
        self.mu0 = mu0      # m²/V·s
        self.vsat = vsat    # m/s
        self.beta = beta

    def caughey_thomas(self, E: torch.Tensor) -> torch.Tensor:
        """μ(E) = μ0 / (1 + (μ0|E|/vsat)^β)^(1/β)"""
        E_abs = torch.abs(E) + 1e-12
        razao = (self.mu0 * E_abs / self.vsat)**self.beta
        return self.mu0 / (1.0 + razao)**(1.0 / self.beta)

    def lombardi(self, E_perp: torch.Tensor, T: float = 300.0) -> torch.Tensor:
        """Modelo simplificado de superfície (Lombardi)."""
        # μ_sr ≈ A / E_perp  (scattering de superfície)
        A = 3e6  # empírica
        mu_sr = A / (torch.abs(E_perp) + 1e3)
        mu_ph = self.mu0 * (T / 300.0)**(-2.0)  # fônons
        return 1.0 / (1.0 / mu_sr + 1.0 / mu_ph)
