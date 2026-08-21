"""
Resíduo da equação de Fokker–Planck:

    ∂p/∂t = −∂/∂x [(a x − b x³) p] + (σ²/2) ∂²p/∂x²
"""

import numpy as np
from typing import Tuple
from .rede_pinn_fp import RedePINN_FP
from .potencial_landau import forca_landau


def residuo_fokker_planck(
    rede: RedePINN_FP,
    X: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
) -> np.ndarray:
    p, pt, px, pxx = rede.derivadas(X)
    x = X[:, 0]
    F = forca_landau(x, a, b)
    # ∂/∂x (F p) ≈ F' p + F p_x ; F' = a − 3 b x²
    Fp_x = (a - 3.0 * b * x ** 2) * p + F * px
    return pt + Fp_x - 0.5 * sigma ** 2 * pxx


def perda_fp(
    rede: RedePINN_FP,
    X_col: np.ndarray,
    X0: np.ndarray,
    p0: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
    peso_pde: float = 1.0,
    peso_ic: float = 10.0,
) -> Tuple[float, float, float]:
    res = residuo_fokker_planck(rede, X_col, a, b, sigma)
    perda_pde = float(np.mean(res ** 2))
    pred0 = rede.prever(X0)
    perda_ic = float(np.mean((pred0 - p0) ** 2))
    return peso_pde * perda_pde + peso_ic * perda_ic, perda_pde, perda_ic
