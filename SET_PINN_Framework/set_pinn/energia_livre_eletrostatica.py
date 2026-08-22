# -*- coding: utf-8 -*-
"""
Módulo 03: Energia Livre de Gibbs e Elipses de Estabilidade
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Tuple
from .constantes_fisicas import e, DTYPE, DEVICE
from .configuracao_dispositivo import ConfiguracaoSET

def delta_F_pm(
    n: torch.Tensor,
    V_D: torch.Tensor,
    V_G: torch.Tensor,
    cfg: ConfiguracaoSET,
    eletrodo: str = "D"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Variação de energia livre na transição n → n±1.
    ΔF_i^±(n, V_D, V_G) = (e/C_Σ) [ e/2 ± (n e - Q_G) ∓ (C_Σ - C_i) V_i ± Σ_{j≠i} C_j V_j ]
    """
    Q_G = cfg.C_G * V_G
    C_Sigma = cfg.C_Sigma
    e_over_C = e / C_Sigma

    if eletrodo.upper() == "D":
        C_i = cfg.C_D
        V_i = V_D
        termo_extra = cfg.C_G * V_G
    else:
        C_i = cfg.C_S
        V_i = torch.zeros_like(V_D)
        termo_extra = cfg.C_D * V_D + cfg.C_G * V_G

    delta_base = (e / 2.0) + (n * e - Q_G)
    delta_F_mais = e_over_C * (delta_base - (C_Sigma - C_i) * V_i + termo_extra)
    delta_F_menos = e_over_C * (-delta_base - (C_Sigma - C_i) * V_i + termo_extra)

    return delta_F_mais, delta_F_menos

def superficie_estabilidade(
    V_D: torch.Tensor,
    V_G: torch.Tensor,
    cfg: ConfiguracaoSET,
    n: int = 0
) -> torch.Tensor:
    """Identifica regiões de estabilidade de Coulomb (elipses) via sinal de ΔF."""
    n_t = torch.full_like(V_D, float(n), dtype=DTYPE, device=DEVICE)
    dF_D_mais, dF_D_menos = delta_F_pm(n_t, V_D, V_G, cfg, "D")
    dF_S_mais, dF_S_menos = delta_F_pm(n_t, V_D, V_G, cfg, "S")
    estavel = (dF_D_mais > 0) & (dF_D_menos > 0) & (dF_S_mais > 0) & (dF_S_menos > 0)
    return estavel.to(DTYPE)
