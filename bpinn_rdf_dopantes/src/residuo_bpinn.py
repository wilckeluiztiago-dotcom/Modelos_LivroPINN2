"""
Resíduo de Poisson e evidência Bayesiana (ELBO) para B-PINN.
"""

import numpy as np
from typing import Tuple
from .rede_bpinn import RedeBayesiana


def laplaciano_rede(rede: RedeBayesiana, x: np.ndarray, pesos, vieses, eps: float = 1e-4) -> np.ndarray:
    x = np.atleast_1d(x).astype(float)
    return (
        rede.forward(x + eps, pesos, vieses)
        - 2 * rede.forward(x, pesos, vieses)
        + rede.forward(x - eps, pesos, vieses)
    ) / (eps ** 2)


def residuo_poisson(
    rede: RedeBayesiana,
    x: np.ndarray,
    rho: np.ndarray,
    pesos,
    vieses,
    epsilon: float = 1.0,
) -> np.ndarray:
    """−ε φ'' − ρ  (deve ≈ 0)."""
    lap = laplaciano_rede(rede, x, pesos, vieses)
    return -epsilon * lap - rho


def elbo_bpinn(
    rede: RedeBayesiana,
    x_col: np.ndarray,
    rho: np.ndarray,
    x_bc: np.ndarray,
    v_bc: np.ndarray,
    n_mc: int = 2,
    sigma_lik: float = 1.0,
    peso_kl: float = 1e-4,
    peso_bc: float = 10.0,
) -> Tuple[float, float, float]:
    """
    −ELBO ≈ MSE(resíduo) + peso_bc MSE(BC) + peso_kl KL(q||p)

    (maximizar ELBO = minimizar esta perda).
    """
    perda_pde = 0.0
    perda_bc = 0.0
    for _ in range(n_mc):
        pesos, vieses = rede.amostrar_pesos()
        res = residuo_poisson(rede, x_col, rho, pesos, vieses)
        perda_pde += float(np.mean(res ** 2))
        pred_bc = rede.forward(x_bc, pesos, vieses)
        perda_bc += float(np.mean((pred_bc - v_bc) ** 2))
    perda_pde /= n_mc
    perda_bc /= n_mc
    kl = rede.kl_prior()
    total = perda_pde / (sigma_lik ** 2) + peso_bc * perda_bc + peso_kl * kl
    return total, perda_pde, kl
