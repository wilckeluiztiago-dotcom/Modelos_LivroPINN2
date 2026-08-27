# =============================================================================
# Módulo 09: Geomecânica do Poço e Fraturamento Hidráulico Inteligente
# Autor: Luiz Tiago Wilcke
# Capítulo 9 do livro
# =============================================================================
"""Poroelasticidade de Biot, Mohr-Coulomb, Mogi-Alamy, Kirsch, fraturamento."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from ..config.configuracoes import FISICA, PINN
from ..utils.utilitarios import gradiente_autograd, LOGGER
from .modulo01_fundamentos import RedeBasePINN

class GeomecanicaPoroelastica:
    """Teoria de Biot e critérios de falha (Cap. 9)."""

    def __init__(self, young: float = 20e9, poisson: float = 0.25, biot: float = 0.8):
        self.E = young
        self.nu = poisson
        self.alpha_biot = biot
        self.G = young / (2*(1+poisson))
        self.lam = young * poisson / ((1+poisson)*(1-2*poisson))
        LOGGER.info("GeomecanicaPoroelastica (Biot) - Luiz Tiago Wilcke")

    def tensao_efetiva(self, tensao_total: float, pressao_poro: float) -> float:
        """σ' = σ - α P (Biot)."""
        return tensao_total - self.alpha_biot * pressao_poro

    def deformacao_linear(self, tensao_efetiva: float) -> float:
        """ε = σ' / E (simplificado uniaxial)."""
        return tensao_efetiva / self.E

    def criterio_mohr_coulomb(self, sigma1: float, sigma3: float, c: float = 5e6,
                              phi_graus: float = 30.0) -> float:
        """Fator de segurança Mohr-Coulomb 3D. FS > 1 = estável."""
        phi = np.deg2rad(phi_graus)
        return (2*c*np.cos(phi) + (sigma1+sigma3)*np.sin(phi)) / ((sigma1-sigma3) + 1e-6)

    def criterio_mogi_alamy(self, sigma1: float, sigma2: float, sigma3: float,
                            a: float = 1.0, b: float = 0.5) -> float:
        """Critério triaxial verdadeiro de Mogi-Alamy (simplificado)."""
        tau_oct = (1/3)*np.sqrt((sigma1-sigma2)**2 + (sigma2-sigma3)**2 + (sigma3-sigma1)**2)
        sigma_m = (sigma1+sigma2+sigma3)/3
        return a + b * sigma_m - tau_oct

    def concentracao_kirsch(self, sigma_h: float, sigma_H: float, theta: float,
                            pressao_poco: float, rw: float = 0.1) -> Tuple[float, float]:
        """Tensões ao redor do poço (Kirsch). θ em radianos."""
        # σ_θθ e σ_rr na parede (r=rw)
        sigma_rr = pressao_poco
        sigma_tt = (sigma_h + sigma_H) - 2*(sigma_H - sigma_h)*np.cos(2*theta) - pressao_poco
        return sigma_rr, sigma_tt

    def janela_pressao_operacional(self, sigma_h: float, sigma_H: float, sigma_v: float,
                                   p_poro: float, tensao_tracao: float = 0.0) -> Dict:
        """Calcula limites de fratura e colapso."""
        # Simplificado
        p_frac = 3*sigma_h - sigma_H - p_poro + tensao_tracao
        p_colapso = (3*sigma_H - sigma_h)/2  # aproximado
        return {"pressao_fratura_Pa": p_frac, "pressao_colapso_Pa": p_colapso,
                "janela_MPa": (p_frac - p_colapso)/1e6}

    def residuo_equilibrio_biot(self, u: torch.Tensor, x: torch.Tensor,
                                 p: torch.Tensor) -> torch.Tensor:
        """Resíduo de equilíbrio mecânico poroelástico simplificado."""
        # div(σ') = 0  →  (λ+2G) u_xx - α p_x ≈ 0
        u_x = gradiente_autograd(u, x)
        u_xx = gradiente_autograd(u_x, x)
        p_x = gradiente_autograd(p, x)
        return (self.lam + 2*self.G) * u_xx - self.alpha_biot * p_x

class PINNGeomecanica(RedeBasePINN):
    def __init__(self):
        super().__init__(dim_entrada=2, dim_saida=2)  # (x,t) -> (u, p)
        self.fisica = GeomecanicaPoroelastica()

    def perda_fisica(self, x, t):
        entrada = torch.cat([x, t], dim=1)
        saida = self.forward(entrada)
        u, p = saida[:,0:1], saida[:,1:2]
        residuo = self.fisica.residuo_equilibrio_biot(u, x, p)
        return torch.mean(residuo**2)
