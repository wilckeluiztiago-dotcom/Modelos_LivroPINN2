"""
Processo de difusão com saltos de Poisson (modelo de Merton / TAT).
Capítulo 17 & Apêndice A.7.
"""

import numpy as np
from typing import Optional, Dict


def densidade_salto_lognormal(eta: np.ndarray, mu_j: float = -0.1, sig_j: float = 0.3) -> np.ndarray:
    """g(η) lognormal para amplitude de salto η > 0."""
    eta = np.maximum(eta, 1e-12)
    return (
        1.0 / (eta * sig_j * np.sqrt(2 * np.pi))
        * np.exp(-0.5 * ((np.log(eta) - mu_j) / sig_j) ** 2)
    )


def passo_jump_diffusion(
    S: float,
    dt: float,
    r: float = 0.05,
    sigma: float = 0.25,
    lam: float = 0.8,
    mu_j: float = -0.1,
    sig_j: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    dS/S = (r − λκ) dt + σ dW + (η − 1) dN

    κ = E[η − 1]; N ~ Poisson(λ dt).
    """
    if rng is None:
        rng = np.random.default_rng()
    kappa = np.exp(mu_j + 0.5 * sig_j ** 2) - 1.0
    dW = rng.normal(0.0, np.sqrt(dt))
    # salto?
    if rng.random() < lam * dt:
        eta = float(rng.lognormal(mu_j, sig_j))
        S = S * eta
    S = S * np.exp((r - lam * kappa - 0.5 * sigma ** 2) * dt + sigma * dW)
    return float(max(S, 1e-8))


def simular_trajetorias(
    n_traj: int = 200,
    n_passos: int = 200,
    dt: float = 0.01,
    S0: float = 1.0,
    r: float = 0.05,
    sigma: float = 0.25,
    lam: float = 0.8,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    g = np.random.default_rng(semente)
    traj = np.zeros((n_traj, n_passos + 1))
    traj[:, 0] = S0
    for i in range(n_traj):
        S = S0
        for k in range(n_passos):
            S = passo_jump_diffusion(S, dt, r, sigma, lam, rng=g)
            traj[i, k + 1] = S
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "S": traj}
