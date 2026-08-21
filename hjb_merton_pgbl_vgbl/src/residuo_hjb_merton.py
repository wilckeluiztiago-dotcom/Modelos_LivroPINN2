"""
Resíduo HJB-Merton:

∂v/∂t + sup_{π,c}{ [r+π(μ−r)] x v_x − c v_x + ½ π² σ² x² v_xx + u(c) } − ρ v = 0
"""

import numpy as np
from typing import Tuple
from .rede_pinn_hjb import RedePINN_HJB
from .merton_crra import utilidade_crra, pi_otimo_merton, c_otimo_aprox


def residuo_hjb(
    rede: RedePINN_HJB,
    X: np.ndarray,
    mu: float = 0.10,
    r: float = 0.08,
    sigma: float = 0.15,
    gamma: float = 2.0,
    rho: float = 0.04,
) -> np.ndarray:
    v, vt, vx, vxx = rede.derivadas(X)
    x = np.maximum(X[:, 1], 1e-4)
    pi = np.clip(pi_otimo_merton(mu, r, sigma, gamma), 0.0, 1.5)
    c = np.array([c_otimo_aprox(xi, gamma, rho, r, mu, sigma) for xi in x])
    u = utilidade_crra(c, gamma)
    drift = (r + pi * (mu - r)) * x * vx - c * vx + 0.5 * (pi ** 2) * (sigma ** 2) * (x ** 2) * vxx
    return vt + drift + u - rho * v


def perda_hjb(
    rede: RedePINN_HJB,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    peso_pde: float = 1.0,
    peso_term: float = 8.0,
    **kwargs,
) -> Tuple[float, float, float]:
    res = residuo_hjb(rede, X_col, **kwargs)
    perda_pde = float(np.mean(res ** 2))
    pred = rede.prever(X_term)
    perda_term = float(np.mean((pred - V_term) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term
