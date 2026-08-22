# -*- coding: utf-8 -*-
"""
Módulo 05: Taxas de Tunelamento Quântico Γ±
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Tuple
from .constantes_fisicas import e, DTYPE
from .configuracao_dispositivo import ConfiguracaoSET
from .energia_livre_eletrostatica import delta_F_pm
from .estatistica_fermi_dirac import integrando_tunelamento

def taxas_tunelamento(
    n: torch.Tensor,
    V_D: torch.Tensor,
    V_G: torch.Tensor,
    cfg: ConfiguracaoSET
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Γ_i^±(n) = (1/(e² R_T^{(i)})) * [-ΔF_i^± / (1 - exp(ΔF_i^± / k_B T_e))]
    Retorna: Γ_S^+, Γ_S^-, Γ_D^+, Γ_D^-
    """
    dF_S_mais, dF_S_menos = delta_F_pm(n, V_D, V_G, cfg, "S")
    dF_D_mais, dF_D_menos = delta_F_pm(n, V_D, V_G, cfg, "D")

    fator_S = 1.0 / (e**2 * cfg.R_T_S)
    fator_D = 1.0 / (e**2 * cfg.R_T_D)

    Gamma_S_mais = fator_S * integrando_tunelamento(dF_S_mais, cfg.T_e)
    Gamma_S_menos = fator_S * integrando_tunelamento(dF_S_menos, cfg.T_e)
    Gamma_D_mais = fator_D * integrando_tunelamento(dF_D_mais, cfg.T_e)
    Gamma_D_menos = fator_D * integrando_tunelamento(dF_D_menos, cfg.T_e)

    # Taxas negativas fisicamente proibidas → zero
    Gamma_S_mais = torch.clamp(Gamma_S_mais, min=0.0)
    Gamma_S_menos = torch.clamp(Gamma_S_menos, min=0.0)
    Gamma_D_mais = torch.clamp(Gamma_D_mais, min=0.0)
    Gamma_D_menos = torch.clamp(Gamma_D_menos, min=0.0)

    return Gamma_S_mais, Gamma_S_menos, Gamma_D_mais, Gamma_D_menos
