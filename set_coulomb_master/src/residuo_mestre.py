"""
Resíduo da equação mestre:

  ∂P(N)/∂t − Σ [Γ(N'→N) P(N') − Γ(N→N') P(N)] = 0
"""

import numpy as np
from typing import Tuple
from .rede_pinn_mestre import RedePINN_Mestre
from .fisica_set import taxas_tunelamento


def residuo_equacao_mestre(
    rede: RedePINN_Mestre,
    N_vals: np.ndarray,
    t_vals: np.ndarray,
    V_g: float,
    V_sd: float,
    N_max_abs: float = 5.0,
    **kwargs,
) -> np.ndarray:
    """
    Para cada (N_i, t_k), calcula ∂P/∂t − fluxo líquido de probabilidade.
    """
    residuos = []
    for t in t_vals:
        P = rede.prever_normalizado(N_vals, t, N_max_abs)
        # ∂P/∂t em cada N
        pts = np.column_stack([N_vals / N_max_abs, np.full_like(N_vals, t, dtype=float)])
        dPdt = rede.derivada_t(pts)
        # renormaliza derivada de forma aproximada
        fluxo = np.zeros_like(P)
        n = len(N_vals)
        for i, N in enumerate(N_vals):
            ga, gr = taxas_tunelamento(int(N), V_g, V_sd, **kwargs)
            fluxo[i] -= (ga + gr) * P[i]
            if i > 0:
                ga_m, _ = taxas_tunelamento(int(N_vals[i - 1]), V_g, V_sd, **kwargs)
                fluxo[i] += ga_m * P[i - 1]
            if i < n - 1:
                _, gr_p = taxas_tunelamento(int(N_vals[i + 1]), V_g, V_sd, **kwargs)
                fluxo[i] += gr_p * P[i + 1]
        residuos.append(dPdt - fluxo)
    return np.concatenate(residuos)


def perda_mestre(
    rede: RedePINN_Mestre,
    N_vals: np.ndarray,
    t_col: np.ndarray,
    P0: np.ndarray,
    V_g: float,
    V_sd: float,
    peso_pde: float = 1.0,
    peso_ic: float = 10.0,
    **kwargs,
) -> Tuple[float, float, float]:
    res = residuo_equacao_mestre(rede, N_vals, t_col, V_g, V_sd, **kwargs)
    perda_pde = float(np.mean(res ** 2))
    P0_pred = rede.prever_normalizado(N_vals, 0.0)
    perda_ic = float(np.mean((P0_pred - P0) ** 2))
    return peso_pde * perda_pde + peso_ic * perda_ic, perda_pde, perda_ic
