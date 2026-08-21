"""
Dinâmica do PLD com reversão à média sazonal e saltos de hidrologia (ENA).
Mercado Livre de Energia (ACL/CCEE).
"""

import numpy as np
from typing import Optional, Dict


def theta_sazonal(t: float, base: float = 4.5, amp: float = 0.4, fase: float = 0.0) -> float:
    """
    Nível médio sazonal de ln(PLD):
        θ(t) = base + amp · sin(2π t + fase)
    (t em anos; base ~ ln(R$/MWh)).
    """
    return base + amp * np.sin(2.0 * np.pi * t + fase)


def passo_pld(
    S: float,
    t: float,
    dt: float,
    k: float = 2.0,
    sigma: float = 0.6,
    lam: float = 1.5,
    mu_j: float = 0.3,
    sig_j: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    d ln S = k (θ(t) − ln S) dt + σ dW + jump (hidrologia)

    Saltos positivos/negativos modelam choques de afluência (ENA).
    """
    if rng is None:
        rng = np.random.default_rng()
    x = np.log(max(S, 1e-6))
    th = theta_sazonal(t)
    dW = rng.normal(0.0, np.sqrt(dt))
    x = x + k * (th - x) * dt + sigma * dW
    if rng.random() < lam * dt:
        # salto de hidrologia (seca → PLD sobe; cheia → cai)
        x = x + rng.normal(mu_j, sig_j)
    return float(np.exp(x))


def simular_pld(
    n_passos: int = 1000,
    dt: float = 0.01,
    S0: float = 150.0,
    semente: Optional[int] = 42,
    **kwargs,
) -> Dict[str, np.ndarray]:
    g = np.random.default_rng(semente)
    S = np.zeros(n_passos + 1)
    S[0] = S0
    t = np.arange(n_passos + 1) * dt
    for i in range(n_passos):
        S[i + 1] = passo_pld(S[i], t[i], dt, rng=g, **kwargs)
    return {"t": t, "S": S, "theta": np.exp([theta_sazonal(ti) for ti in t])}
