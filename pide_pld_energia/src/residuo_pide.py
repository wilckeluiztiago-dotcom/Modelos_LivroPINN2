"""
Resíduo da PIDE:

∂V/∂t + ½ σ² S² V_SS + k(θ(t)−ln S) S V_S − r V
  + λ ∫ [V(Sη,t) − V(S,t)] g(η) dη = 0

Integral por Monte Carlo contínuo.
"""

import numpy as np
from typing import Tuple, Optional
from .rede_pinn_pide import RedePINN_PIDE
from .pld_hidrologia import theta_sazonal


def integral_salto_mc(
    rede: RedePINN_PIDE,
    S: np.ndarray,
    t: np.ndarray,
    n_mc: int = 12,
    mu_j: float = 0.3,
    sig_j: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    S = np.atleast_1d(S)
    t = np.atleast_1d(t)
    acc = np.zeros_like(S, dtype=float)
    V0 = rede.prever(np.column_stack([S, t]))
    for _ in range(n_mc):
        eta = np.exp(rng.normal(mu_j, sig_j, size=S.shape))
        Vj = rede.prever(np.column_stack([S * eta, t]))
        acc += Vj - V0
    return acc / n_mc


def residuo_pide_energia(
    rede: RedePINN_PIDE,
    X: np.ndarray,
    k: float = 2.0,
    sigma: float = 0.6,
    r: float = 0.08,
    lam: float = 1.5,
    n_mc: int = 10,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    V, Vt, Vs, Vss = rede.derivadas(X)
    S = np.maximum(X[:, 0], 1e-4)
    t = X[:, 1]
    th = np.array([theta_sazonal(ti) for ti in t])
    drift = k * (th - np.log(S)) * S
    integ = integral_salto_mc(rede, S, t, n_mc=n_mc, rng=rng)
    return Vt + 0.5 * sigma ** 2 * S ** 2 * Vss + drift * Vs - r * V + lam * integ


def perda_pide(
    rede: RedePINN_PIDE,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    peso_pde: float = 1.0,
    peso_term: float = 12.0,
    **kwargs,
) -> Tuple[float, float, float]:
    rng = kwargs.pop("rng", None)
    res = residuo_pide_energia(rede, X_col, rng=rng, **kwargs)
    perda_pde = float(np.mean(res ** 2))
    pred = rede.prever(X_term)
    perda_term = float(np.mean((pred - V_term) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term
