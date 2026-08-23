"""
Módulo: Tunelamento (WKB / BTBT)
Autor: Luiz Tiago Wilcke
"""

import torch
import math
from parametros_materiais_si import ParametrosSilicio


class Tunelamento:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.hbar = self.mat.hbar
        self.m = self.mat.massa_efetiva_kg()
        self.q = self.mat.q

    def transmissao_wkb(self, V_barreira_eV: torch.Tensor, L_nm: torch.Tensor) -> torch.Tensor:
        """T ≈ exp(-2κL), κ = sqrt(2mV)/ħ"""
        V = V_barreira_eV * self.q
        L = L_nm * 1e-9
        kappa = torch.sqrt(2 * self.m * torch.clamp(V, min=1e-30)) / self.hbar
        T = torch.exp(-2 * kappa * L)
        return torch.clamp(T, 1e-20, 1.0)

    def corrente_btbt(self, E: torch.Tensor, Eg_eV: float = 1.12) -> torch.Tensor:
        """Corrente de band-to-band tunneling aproximada (Kane)."""
        # forma simplificada
        A = 1e15
        B = 2e7
        E_abs = torch.abs(E) + 1e-6
        return A * E_abs**2 * torch.exp(-B * Eg_eV**1.5 / E_abs)
