# =============================================================================
# Módulo 08: Dinâmica de Escoamento de Fluidos Não-Newtonianos
# Autor: Luiz Tiago Wilcke
# Capítulo 8 do livro
# =============================================================================
"""Lei de Potência, Bingham, Herschel-Bulkley, tixotropia Moore-Hahn-Mewis, Oldroyd-B."""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple
from ..config.configuracoes import FISICA, PINN
from ..utils.utilitarios import gradiente_autograd, LOGGER
from .modulo01_fundamentos import RedeBasePINN

class FluidoNaoNewtoniano:
    """Modelos reológicos clássicos e avançados (Cap. 8)."""

    def __init__(self):
        LOGGER.info("FluidoNaoNewtoniano inicializado - Luiz Tiago Wilcke")

    def viscosidade_lei_potencia(self, taxa_cisalhamento: float, k: float = 0.5, n: float = 0.6) -> float:
        """Modelo Ostwald-de Waele (Lei de Potência). μ_app = K * γ̇^(n-1)"""
        return k * (max(abs(taxa_cisalhamento), 1e-8)**(n-1))

    def tensao_bingham(self, taxa: float, tau0: float = 5.0, mu_p: float = 0.05) -> float:
        """Modelo Bingham Plastic. τ = τ0 + μp * γ̇"""
        return tau0 + mu_p * abs(taxa)

    def tensao_herschel_bulkley(self, taxa: float, tau0: float = 3.0, k: float = 0.4, n: float = 0.5) -> float:
        """Modelo Herschel-Bulkley. τ = τ0 + K * γ̇^n"""
        return tau0 + k * (max(abs(taxa), 1e-8)**n)

    def viscosidade_aparente_hb(self, taxa: float, tau0: float = 3.0, k: float = 0.4, n: float = 0.5) -> float:
        """Viscosidade aparente Herschel-Bulkley."""
        return self.tensao_herschel_bulkley(taxa, tau0, k, n) / max(abs(taxa), 1e-8)

    def estrutura_tixotropica(self, lambda_s: float, taxa: float, a: float = 0.1, b: float = 0.05) -> float:
        """Modelo de Moore-Hahn-Mewis para carga estrutural (tixotropia)."""
        # dλ/dt = a(1-λ) - b*λ*|γ̇|
        return a * (1 - lambda_s) - b * lambda_s * abs(taxa)

    def tensao_oldroyd_b(self, taxa: float, mu_s: float = 0.01, mu_p: float = 0.1,
                         lambda_t: float = 0.5) -> float:
        """Aproximação simplificada de tensão para fluido Oldroyd-B (Weissenberg)."""
        # Em regime permanente uniaxial simplificado
        wi = lambda_t * abs(taxa)
        return (mu_s + mu_p) * taxa / (1 + wi**2 + 1e-8)

    def lei_darcy_modificada_potencia(self, grad_p: float, k_perm: float, k_reol: float,
                                       n: float = 0.6) -> float:
        """Lei de Darcy modificada para fluidos de lei de potência em meios porosos."""
        # u = (k/μ_eff) * |∇P|^(1/n) * sign
        mu_eff = k_reol  # simplificado
        return (k_perm / mu_eff) * (abs(grad_p)**(1/n)) * np.sign(grad_p)

    def residuo_escoamento_anular(self, u: torch.Tensor, r: torch.Tensor,
                                   tau0: float = 3.0, k: float = 0.4, n: float = 0.5) -> torch.Tensor:
        """Resíduo de momento para escoamento anular Herschel-Bulkley."""
        du_dr = gradiente_autograd(u, r)
        # Simplificado: dτ/dr + τ/r = dp/dz (equilíbrio)
        tau = tau0 + k * (torch.abs(du_dr) + 1e-8)**n
        return gradiente_autograd(tau, r) + tau / (r + 1e-8)

class PINNNaoNewtoniano(RedeBasePINN):
    def __init__(self):
        super().__init__(dim_entrada=2, dim_saida=1)
        self.fisica = FluidoNaoNewtoniano()

    def perda_fisica(self, r, z):
        entrada = torch.cat([r, z], dim=1)
        u = self.forward(entrada)
        residuo = self.fisica.residuo_escoamento_anular(u, r)
        return torch.mean(residuo**2)
