"""
Modelo de Heston para ações de alta liquidez (PETR4, VALE3).
Volatilidade estocástica com correlação negativa (skew de commodities).
"""

import numpy as np
from typing import Optional, Dict, Tuple


def passo_heston(
    S: float,
    v: float,
    dt: float,
    r: float = 0.10,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    dS = r S dt + √v S dW1
    dv = κ(θ − v) dt + ξ √v dW2
    corr(dW1, dW2) = ρ < 0
    """
    if rng is None:
        rng = np.random.default_rng()
    v = max(v, 1e-8)
    z1 = rng.normal()
    z2 = rho * z1 + np.sqrt(max(1 - rho ** 2, 0)) * rng.normal()
    S_new = S * np.exp((r - 0.5 * v) * dt + np.sqrt(v * dt) * z1)
    v_new = v + kappa * (theta - v) * dt + xi * np.sqrt(v * dt) * z2
    v_new = max(v_new, 1e-8)
    return float(S_new), float(v_new)


def simular_heston(
    n_passos: int = 500,
    dt: float = 0.002,
    S0: float = 35.0,   # PETR4 ~ R$
    v0: float = 0.06,
    r: float = 0.10,
    kappa: float = 2.0,
    theta: float = 0.04,
    xi: float = 0.5,
    rho: float = -0.7,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    g = np.random.default_rng(semente)
    S = np.zeros(n_passos + 1)
    v = np.zeros(n_passos + 1)
    S[0], v[0] = S0, v0
    for k in range(n_passos):
        S[k + 1], v[k + 1] = passo_heston(
            S[k], v[k], dt, r, kappa, theta, xi, rho, g
        )
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "S": S, "v": v}


def payoff_call(S: np.ndarray, K: float) -> np.ndarray:
    return np.maximum(S - K, 0.0)
