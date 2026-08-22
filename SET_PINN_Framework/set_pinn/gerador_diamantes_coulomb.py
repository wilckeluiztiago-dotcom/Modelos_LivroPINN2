# -*- coding: utf-8 -*-
"""
Módulo 26: Mapeamento Bidimensional dos Diamantes de Coulomb
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .taxas_tunelamento import taxas_tunelamento
from .configuracao_dispositivo import ConfiguracaoSET
from .constantes_fisicas import e, DTYPE, DEVICE

def mapa_corrente(
    cfg: ConfiguracaoSET,
    V_D_range: torch.Tensor,
    V_G_range: torch.Tensor,
    n_max: int = 3
) -> torch.Tensor:
    """
    I(V_D, V_G) = e Σ_n P(n) [Γ_D^+(n) - Γ_D^-(n)]
    (P(n) aproximado por distribuição uniforme para demonstração)
    """
    VD, VG = torch.meshgrid(V_D_range, V_G_range, indexing="ij")
    I = torch.zeros_like(VD, dtype=DTYPE, device=DEVICE)

    for n_val in range(-n_max, n_max + 1):
        n_t = torch.full_like(VD, float(n_val), dtype=DTYPE, device=DEVICE)
        _, _, Gd_mais, Gd_menos = taxas_tunelamento(n_t, VD, VG, cfg)
        I = I + e * (Gd_mais - Gd_menos)

    return I / (2 * n_max + 1)  # normalização aproximada
