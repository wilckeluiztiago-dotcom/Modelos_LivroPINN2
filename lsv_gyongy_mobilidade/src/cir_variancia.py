"""
Processo CIR de variância estocástica ν_t (fator de Heston).
Modela flutuações rápidas de espalhamento fonônico.
"""

import numpy as np
from typing import Optional, Tuple


def passo_cir(
    nu: float,
    dt: float,
    kappa: float = 2.0,
    theta: float = 1.0,
    xi: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Euler–Maruyama refletido para
        dν = κ(θ − ν) dt + ξ √ν dW
    """
    if rng is None:
        rng = np.random.default_rng()
    dW = rng.normal(0.0, np.sqrt(dt))
    nu_new = nu + kappa * (theta - nu) * dt + xi * np.sqrt(max(nu, 0.0)) * dW
    return float(max(nu_new, 1e-8))


def simular_cir(
    n_passos: int,
    dt: float = 0.01,
    nu0: float = 1.0,
    kappa: float = 2.0,
    theta: float = 1.0,
    xi: float = 0.5,
    semente: Optional[int] = 0,
) -> np.ndarray:
    g = np.random.default_rng(semente)
    traj = np.zeros(n_passos + 1)
    traj[0] = nu0
    for k in range(n_passos):
        traj[k + 1] = passo_cir(traj[k], dt, kappa, theta, xi, g)
    return traj
