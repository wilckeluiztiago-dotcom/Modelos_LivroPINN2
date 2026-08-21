"""
Market making Avellaneda–Stoikov para mini-índice (WIN) e mini-dólar (WDO).
"""

import numpy as np
from typing import Optional, Dict, Tuple


def intensidade_chegada(delta: float, A: float = 1.5, k: float = 1.0) -> float:
    """λ(δ) = A exp(−k δ) — taxa de fill em função da distância ao mid."""
    return A * np.exp(-k * max(delta, 0.0))


def reservas_as(
    s: float,
    q: float,
    sigma: float,
    gamma: float,
    T_resto: float,
    k: float = 1.0,
) -> Tuple[float, float]:
    """
    Preço de reserva e spread ótimo (fórmulas AS clássicas):

        r = s − q γ σ² (T−t) − (1/γ) ln(1 + γ/k)
        δ* = (1/γ) ln(1 + γ/k) + ½ γ σ² (T−t)
    """
    half = (1.0 / gamma) * np.log(1.0 + gamma / k) if gamma > 1e-12 else 1.0 / k
    r = s - q * gamma * sigma ** 2 * T_resto - half
    delta = half + 0.5 * gamma * sigma ** 2 * T_resto
    return float(r), float(max(delta, 1e-4))


def simular_mm(
    n_passos: int = 2000,
    dt: float = 0.01,
    s0: float = 100.0,
    sigma: float = 0.5,
    gamma: float = 0.1,
    A: float = 1.5,
    k: float = 1.0,
    q_max: int = 10,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """
    Simula market maker AS:
      - mid price: dS = σ dW
      - quotes: bid = r − δ, ask = r + δ
      - fills Poisson com λ(δ)
    """
    g = np.random.default_rng(semente)
    T = n_passos * dt
    s = s0
    q = 0
    cash = 0.0
    traj_s = np.zeros(n_passos + 1)
    traj_q = np.zeros(n_passos + 1)
    traj_pnl = np.zeros(n_passos + 1)
    traj_s[0] = s
    for step in range(n_passos):
        t_resto = T - step * dt
        r, delta = reservas_as(s, q, sigma, gamma, max(t_resto, 0.01), k)
        bid, ask = r - delta, r + delta
        # fills
        if q < q_max and g.random() < intensidade_chegada(s - bid, A, k) * dt:
            q += 1
            cash -= bid
        if q > -q_max and g.random() < intensidade_chegada(ask - s, A, k) * dt:
            q -= 1
            cash += ask
        # mid
        s = s + sigma * g.normal(0, np.sqrt(dt))
        traj_s[step + 1] = s
        traj_q[step + 1] = q
        traj_pnl[step + 1] = cash + q * s
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "s": traj_s, "q": traj_q, "pnl": traj_pnl}
