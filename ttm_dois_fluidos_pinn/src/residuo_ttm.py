"""
Resíduos PINN do TTM:

  R_e = C_e ∂_t T_e − ∇·(κ_e ∇ T_e) + G(T_e − T_L) − σ|E|²
  R_L = C_L ∂_t T_L − ∇·(κ_L ∇ T_L) − G(T_e − T_L)

Em 1D: ∇·(κ ∇T) = κ ∂_{xx} T (κ constante).
"""

import numpy as np
from typing import Tuple, Dict
from .rede_pinn_ttm import RedePINN_TTM
from .fisica_ttm import parametros_ttm_default, fonte_joule


def residuos_ttm(
    rede: RedePINN_TTM,
    X: np.ndarray,
    p: Dict[str, float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if p is None:
        p = parametros_ttm_default()
    Te, TL, Te_t, TL_t, Te_xx, TL_xx = rede.derivadas(X)
    Joule = fonte_joule(p["E_field"], p["sigma_J"])
    R_e = p["C_e"] * Te_t - p["kappa_e"] * Te_xx + p["G"] * (Te - TL) - Joule
    R_L = p["C_L"] * TL_t - p["kappa_L"] * TL_xx - p["G"] * (Te - TL)
    return R_e, R_L


def perda_ttm(
    rede: RedePINN_TTM,
    X_col: np.ndarray,
    X0: np.ndarray,
    Te0: np.ndarray,
    TL0: np.ndarray,
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_ic: float = 12.0,
) -> Tuple[float, float, float]:
    R_e, R_L = residuos_ttm(rede, X_col, p)
    perda_pde = float(np.mean(R_e ** 2) + np.mean(R_L ** 2))
    pred0 = rede.prever(X0)
    if pred0.ndim == 1:
        pred0 = pred0.reshape(1, -1)
    perda_ic = float(np.mean((pred0[:, 0] - Te0) ** 2) + np.mean((pred0[:, 1] - TL0) ** 2))
    return peso_pde * perda_pde + peso_ic * perda_ic, perda_pde, perda_ic
