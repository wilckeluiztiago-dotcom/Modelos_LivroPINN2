"""
Módulo: Recombinação SRH e Auger
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class Recombinacao:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()
        self.tau_n = self.mat.tau_n_srh
        self.tau_p = self.mat.tau_p_srh
        self.C_n = 2.8e-31   # m⁶/s (Auger)
        self.C_p = 9.9e-32

    def srh(self, n: torch.Tensor, p: torch.Tensor, ni: float = 1e16) -> torch.Tensor:
        """R_SRH = (np - ni²) / (τ_p(n+ni) + τ_n(p+ni))"""
        num = n * p - ni**2
        den = self.tau_p * (n + ni) + self.tau_n * (p + ni) + 1e-30
        return num / den

    def auger(self, n: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        """R_Auger = (C_n n + C_p p)(np - ni²)"""
        ni2 = 1e32  # aproximação
        return (self.C_n * n + self.C_p * p) * (n * p - ni2)

    def total(self, n, p, ni=1e16):
        return self.srh(n, p, ni) + self.auger(n, p)
