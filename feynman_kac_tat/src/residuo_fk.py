"""
Resíduo da PIDE de Feynman–Kac com saltos:

∂V/∂t + (1/2)σ² S² V_SS + (r−λκ) S V_S − (r+λ) V
    + λ ∫ V(S η, t) g(η) dη  = 0

O integral é estimado por Monte Carlo contínuo.
"""

import numpy as np
from typing import Tuple, Optional
from .rede_pinn_fk import RedePINN_FK
from .processo_saltos import densidade_salto_lognormal


def operador_integral_mc(
    rede: RedePINN_FK,
    S: np.ndarray,
    t: np.ndarray,
    n_mc: int = 16,
    mu_j: float = -0.1,
    sig_j: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    ∫ V(S η, t) g(η) dη  ≈ média de V(S η_k, t) com η_k ~ g.
    """
    if rng is None:
        rng = np.random.default_rng()
    S = np.atleast_1d(S)
    t = np.atleast_1d(t)
    integral = np.zeros_like(S, dtype=float)
    for k in range(n_mc):
        eta = rng.lognormal(mu_j, sig_j, size=S.shape)
        pts = np.column_stack([S * eta, t])
        integral += rede.prever(pts)
    return integral / n_mc


def residuo_feynman_kac(
    rede: RedePINN_FK,
    X: np.ndarray,
    r: float = 0.05,
    sigma: float = 0.25,
    lam: float = 0.8,
    mu_j: float = -0.1,
    sig_j: float = 0.3,
    n_mc: int = 12,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    V, Vt, Vs, Vss = rede.derivadas(X)
    S = X[:, 0]
    t = X[:, 1]
    kappa = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
    integ = operador_integral_mc(rede, S, t, n_mc, mu_j, sig_j, rng)
    return (
        Vt
        + 0.5 * sigma ** 2 * S ** 2 * Vss
        + (r - lam * kappa) * S * Vs
        - (r + lam) * V
        + lam * integ
    )


def perda_fk(
    rede: RedePINN_FK,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    r: float = 0.05,
    sigma: float = 0.25,
    lam: float = 0.8,
    peso_pde: float = 1.0,
    peso_term: float = 10.0,
    n_mc: int = 10,
    semente: Optional[int] = None,
) -> Tuple[float, float, float]:
    rng = np.random.default_rng(semente)
    res = residuo_feynman_kac(rede, X_col, r, sigma, lam, n_mc=n_mc, rng=rng)
    perda_pde = float(np.mean(res ** 2))
    pred = rede.prever(X_term)
    perda_term = float(np.mean((pred - V_term) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term
