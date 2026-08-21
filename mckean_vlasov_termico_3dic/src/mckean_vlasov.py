"""
Difusão de McKean–Vlasov: drift acoplado à média da população.
Capítulos 24 & 40 — contágio térmico.
"""

import numpy as np
from typing import Optional, Dict


def passo_mckean_vlasov(
    X: np.ndarray,
    dt: float,
    a: float = 1.0,
    sigma: float = 0.15,
    aquecimento: float = 0.05,
    T_crit: float = 1.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sistema de partículas interagentes:

        dX^i = a (X̄_t − X^i) dt + α 1_{X̄ > T_crit} dt + σ dW^i

    O termo a(X̄ − X^i) atrai cada elemento à média (difusão térmica acoplada).
    O termo α modela auto-aceleração (fuga térmica) quando a média excede T_crit.
    """
    if rng is None:
        rng = np.random.default_rng()
    Xbar = float(np.mean(X))
    drift = a * (Xbar - X)
    if Xbar > T_crit:
        drift = drift + aquecimento
    dW = rng.normal(0.0, np.sqrt(dt), size=X.shape)
    return X + drift * dt + sigma * dW


def simular_populacao(
    n_particulas: int = 200,
    n_passos: int = 1500,
    dt: float = 0.01,
    a: float = 1.2,
    sigma: float = 0.12,
    aquecimento: float = 0.08,
    T_crit: float = 1.2,
    T0_mean: float = 0.8,
    T0_std: float = 0.15,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Simula N nanotransistores (temperaturas locais)."""
    g = np.random.default_rng(semente)
    X = g.normal(T0_mean, T0_std, size=n_particulas)
    X = np.clip(X, 0.1, 3.0)
    traj_mean = np.zeros(n_passos + 1)
    traj_std = np.zeros(n_passos + 1)
    traj_mean[0] = X.mean()
    traj_std[0] = X.std()
    for k in range(n_passos):
        X = passo_mckean_vlasov(X, dt, a, sigma, aquecimento, T_crit, g)
        X = np.clip(X, 0.05, 4.0)
        traj_mean[k + 1] = X.mean()
        traj_std[k + 1] = X.std()
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "mean": traj_mean, "std": traj_std, "X_final": X}
