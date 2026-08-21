"""
Resíduo da EDP de Heston:

∂V/∂τ = ½ v S² V_SS + ρ ξ v S V_Sv + ½ ξ² v V_vv
       + r S V_S + κ(θ−v) V_v − r V

(τ = tempo até vencimento; sinal conforme convenção τ↓0 → payoff)
"""

import numpy as np
from typing import Tuple
from .rede_pinn_heston import RedePINN_Heston


def residuo_heston(
    rede: RedePINN_Heston,
    X: np.ndarray,
    r: float = 0.10,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
) -> np.ndarray:
    V, VS, Vv, Vtau, VSS, VSv, Vvv = rede.derivadas(X)
    S = np.maximum(X[:, 0], 1e-4)
    v = np.maximum(X[:, 1], 1e-6)
    # forma: V_τ − L[V] = 0  com L o operador espacial
    L = (
        0.5 * v * S ** 2 * VSS
        + rho * xi * v * S * VSv
        + 0.5 * xi ** 2 * v * Vvv
        + r * S * VS
        + kappa * (theta - v) * Vv
        - r * V
    )
    return Vtau - L


def perda_heston(
    rede: RedePINN_Heston,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    peso_pde: float = 1.0,
    peso_term: float = 10.0,
    **kwargs,
) -> Tuple[float, float, float]:
    res = residuo_heston(rede, X_col, **kwargs)
    perda_pde = float(np.mean(res ** 2))
    pred = rede.prever(X_term)
    perda_term = float(np.mean((pred - V_term) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term
