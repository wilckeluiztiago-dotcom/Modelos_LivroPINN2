"""
Dinâmica de Langevin e equação de Fokker–Planck associada.
"""

import numpy as np
from typing import Optional, Dict
from .potencial_landau import forca_landau


def passo_langevin(
    x: float,
    dt: float,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    dX = (a X − b X³) dt + σ dW
    """
    if rng is None:
        rng = np.random.default_rng()
    F = forca_landau(np.array([x]), a, b)[0]
    dW = rng.normal(0.0, np.sqrt(dt))
    return float(x + F * dt + sigma * dW)


def simular_langevin(
    n_passos: int = 5000,
    dt: float = 0.01,
    x0: float = -1.0,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    g = np.random.default_rng(semente)
    traj = np.zeros(n_passos + 1)
    traj[0] = x0
    for k in range(n_passos):
        traj[k + 1] = passo_langevin(traj[k], dt, a, b, sigma, g)
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "x": traj}


def densidade_estacionaria_analitica(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
) -> np.ndarray:
    """
    p_∞(x) ∝ exp(−2 V(x) / σ²)
    """
    from .potencial_landau import potencial_landau
    V = potencial_landau(x, a, b)
    p = np.exp(-2.0 * V / (sigma ** 2))
    # normaliza
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    p = p / (np.trapezoid(p, dx=dx) + 1e-15)
    return p
