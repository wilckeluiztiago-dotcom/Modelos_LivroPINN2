"""
Módulo: Tunelamento WKB e Band-to-Band (Kane)
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

    def transmissao_wkb(self, V_eV: torch.Tensor, L_nm: torch.Tensor) -> torch.Tensor:
        """T = exp(-2 ∫ κ dx) ≈ exp(-2κL) para barreira retangular."""
        V = torch.clamp(V_eV * self.q, min=1e-25)
        L = L_nm * 1e-9
        kappa = torch.sqrt(2.0 * self.m * V) / self.hbar
        T = torch.exp(-2.0 * kappa * L)
        return torch.clamp(T, 1e-25, 1.0)

    def corrente_btbt_kane(self, E: torch.Tensor, Eg_eV: float = 1.12) -> torch.Tensor:
        """
        Modelo de Kane para BTBT:
          G = A · E² · exp(-B · Eg^{3/2} / |E|)
        Unidades: E em V/m, G em m⁻³ s⁻¹ (taxa de geração).
        """
        A = 4.0e16          # valor típico Si
        B = 1.9e9           # V/m · eV^{-3/2}
        E_abs = torch.abs(E) + 1e3
        return A * E_abs**2 * torch.exp(-B * (Eg_eV**1.5) / E_abs)
