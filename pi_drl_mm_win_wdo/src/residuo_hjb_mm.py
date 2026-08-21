"""
Resíduo HJB Avellaneda–Stoikov para regularizar o Critic:

  ∂v/∂t + ½ γ² σ² q² v
  + max_{δ^b}[ λ^b (v(q+1) e^{γ δ^b} − v) ]
  + max_{δ^a}[ λ^a (v(q−1) e^{γ δ^a} − v) ] ≈ 0
"""

import numpy as np
from typing import Tuple
from .rede_pidrl import Critic
from .avellaneda_stoikov import intensidade_chegada


def residuo_hjb_as(
    critic: Critic,
    estados: np.ndarray,
    sigma: float = 0.5,
    gamma: float = 0.1,
    A: float = 1.5,
    k: float = 1.0,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    estados: (N, 3) = (t_norm, s_norm, q_norm)
    """
    estados = np.asarray(estados, dtype=float)
    if estados.ndim == 1:
        estados = estados.reshape(1, -1)
    v = critic.valor(estados)
    # ∂v/∂t
    ep = estados.copy(); em = estados.copy()
    ep[:, 0] += eps; em[:, 0] -= eps
    vt = (critic.valor(ep) - critic.valor(em)) / (2 * eps)

    q = estados[:, 2] * 10.0  # desnormaliza approx
    # termo de inventário
    inv = 0.5 * gamma ** 2 * sigma ** 2 * (q ** 2) * v

    # max over δ approx com δ* AS
    delta_star = (1.0 / gamma) * np.log(1.0 + gamma / k) if gamma > 1e-12 else 0.5
    delta_star = max(delta_star, 0.05)
    # v(q+1), v(q-1)
    eqp = estados.copy(); eqp[:, 2] = np.clip(estados[:, 2] + 0.1, -1, 1)
    eqm = estados.copy(); eqm[:, 2] = np.clip(estados[:, 2] - 0.1, -1, 1)
    v_qp = critic.valor(eqp)
    v_qm = critic.valor(eqm)
    lam = intensidade_chegada(delta_star, A, k)
    term_b = lam * (v_qp * np.exp(gamma * delta_star) - v)
    term_a = lam * (v_qm * np.exp(gamma * delta_star) - v)

    return vt + inv + term_b + term_a


def perda_critic_pidrl(
    critic: Critic,
    estados: np.ndarray,
    alvos: np.ndarray,
    peso_td: float = 1.0,
    peso_hjb: float = 0.3,
    **kwargs,
) -> Tuple[float, float, float]:
    pred = critic.valor(estados)
    perda_td = float(np.mean((pred - alvos) ** 2))
    res = residuo_hjb_as(critic, estados, **kwargs)
    perda_hjb = float(np.mean(res ** 2))
    return peso_td * perda_td + peso_hjb * perda_hjb, perda_td, perda_hjb
