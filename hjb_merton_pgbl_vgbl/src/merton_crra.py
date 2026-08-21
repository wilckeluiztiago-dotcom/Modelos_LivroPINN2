"""
Modelo de Merton com utilidade CRRA para previdência PGBL/VGBL.
"""

import numpy as np
from typing import Tuple


def utilidade_crra(c: np.ndarray, gamma: float = 2.0) -> np.ndarray:
    """u(c) = c^{1-γ} / (1-γ), γ ≠ 1."""
    c = np.maximum(c, 1e-8)
    if abs(gamma - 1.0) < 1e-8:
        return np.log(c)
    return (c ** (1.0 - gamma)) / (1.0 - gamma)


def pi_otimo_merton(
    mu: float,
    r: float,
    sigma: float,
    gamma: float,
) -> float:
    """
    Fração ótima em risco (Merton clássico, c endógeno separado):
        π* = (μ − r) / (γ σ²)
    """
    if sigma < 1e-8 or gamma < 1e-8:
        return 0.0
    return float((mu - r) / (gamma * sigma ** 2))


def c_otimo_aprox(
    x: float,
    gamma: float,
    rho: float,
    r: float,
    mu: float,
    sigma: float,
) -> float:
    """
    Consumo ótimo aproximado (Merton infinito):
        c* = (ρ − (1-γ)·(r + (μ-r)²/(2γσ²))) / γ  ·  x
    (taxa efetiva de depleção).
    """
    excess = (mu - r) ** 2 / (2.0 * gamma * sigma ** 2 + 1e-12)
    num = rho - (1.0 - gamma) * (r + excess)
    taxa = max(num / gamma, 0.01)
    return float(taxa * max(x, 0.0))


def simular_riqueza(
    n_passos: int = 360,
    dt: float = 1.0 / 12.0,  # mensal
    x0: float = 100.0,
    aporte: float = 1.0,    # aporte mensal programado
    mu: float = 0.10,
    r: float = 0.08,        # CDI aproximado
    sigma: float = 0.15,
    gamma: float = 2.0,
    rho: float = 0.04,
    semente: int = 42,
) -> dict:
    """Simula trajetória de riqueza sob política Merton + aportes."""
    g = np.random.default_rng(semente)
    pi = np.clip(pi_otimo_merton(mu, r, sigma, gamma), 0.0, 1.0)
    x = x0
    traj = np.zeros(n_passos + 1)
    traj[0] = x
    for k in range(n_passos):
        c = c_otimo_aprox(x, gamma, rho, r, mu, sigma) * dt
        dW = g.normal(0, np.sqrt(dt))
        dx = (r + pi * (mu - r)) * x * dt - c + aporte * dt
        dx += pi * sigma * x * dW
        x = max(x + dx, 0.0)
        traj[k + 1] = x
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "x": traj, "pi": pi}
