"""
Resíduo da EDP de McKean–Vlasov:

    ∂p/∂t + a ∂/∂x [(X̄ − x) p] − (σ²/2) ∂²p/∂x² = 0

onde X̄_t = ∫ x p(x,t) dx  (média populacional).
"""

import numpy as np
from typing import Tuple
from .rede_pinn_mv import RedePINN_MV


def media_populacional(
    rede: RedePINN_MV,
    t: float,
    x_grade: np.ndarray,
) -> float:
    """X̄(t) ≈ ∫ x p_θ(x,t) dx / ∫ p_θ dx  (integração contínua)."""
    pts = np.column_stack([x_grade, np.full_like(x_grade, t)])
    p = rede.prever(pts)
    p = np.maximum(p, 0.0)
    massa = np.trapezoid(p, x_grade)
    if massa < 1e-12:
        return float(np.mean(x_grade))
    return float(np.trapezoid(x_grade * p, x_grade) / massa)


def residuo_mckean_vlasov(
    rede: RedePINN_MV,
    X: np.ndarray,
    x_grade: np.ndarray,
    a: float = 1.2,
    sigma: float = 0.12,
) -> np.ndarray:
    p, pt, px, pxx = rede.derivadas(X)
    # X̄ por tempo (usa t médio do batch ou por ponto)
    # aproximação: um X̄ global no batch pelo t médio
    t_med = float(np.mean(X[:, 1]))
    Xbar = media_populacional(rede, t_med, x_grade)
    x = X[:, 0]
    # ∂/∂x [(X̄ − x) p] = −p + (X̄ − x) p_x
    div = -p + (Xbar - x) * px
    return pt + a * div - 0.5 * sigma ** 2 * pxx


def perda_mv(
    rede: RedePINN_MV,
    X_col: np.ndarray,
    X0: np.ndarray,
    p0: np.ndarray,
    x_grade: np.ndarray,
    a: float = 1.2,
    sigma: float = 0.12,
    peso_pde: float = 1.0,
    peso_ic: float = 12.0,
) -> Tuple[float, float, float]:
    res = residuo_mckean_vlasov(rede, X_col, x_grade, a, sigma)
    perda_pde = float(np.mean(res ** 2))
    pred0 = rede.prever(X0)
    perda_ic = float(np.mean((pred0 - p0) ** 2))
    return peso_pde * perda_pde + peso_ic * perda_ic, perda_pde, perda_ic
