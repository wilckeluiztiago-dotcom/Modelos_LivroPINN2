"""
Módulo: Transporte Balístico (Top-of-Barrier / Landauer simplificado)
Autor: Luiz Tiago Wilcke
"""

import torch
import math
from parametros_materiais_si import ParametrosSilicio


class TransporteBalistico:
    def __init__(self, mat: ParametrosSilicio = None, W_nm: float = 15.0):
        self.mat = mat or ParametrosSilicio()
        self.W = W_nm * 1e-9  # metros
        self.q = self.mat.q
        self.h = self.mat.h

    def corrente_landauer(self, T_barreira: float, Vds: float, T: float = 300.0):
        """
        I ≈ (2q/h) ∫ T(E) [f_S - f_D] dE  (aproximação top-of-barrier)
        T_barreira: transmissão efetiva (0-1)
        """
        # aproximação linear de resposta
        G0 = (2 * self.q**2 / self.h) * self.W   # conductance quantum escalada
        I = G0 * T_barreira * Vds
        return I

    def transmissao_wkb(self, altura_barreira_eV: float, largura_nm: float, m_star=None):
        """Transmissão WKB aproximada para barreira triangular/retangular."""
        if m_star is None:
            m_star = self.mat.massa_efetiva_kg()
        hbar = self.mat.hbar
        L = largura_nm * 1e-9
        V = altura_barreira_eV * self.q
        # κ ≈ sqrt(2m*V)/ħ
        kappa = math.sqrt(2 * m_star * max(V, 1e-21)) / hbar
        T = math.exp(-2 * kappa * L)
        return min(max(T, 1e-12), 1.0)
