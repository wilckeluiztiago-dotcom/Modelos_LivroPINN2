"""
Dinâmica de carga em Single-Electron Transistor / memória de elétron único.
q ∈ Z (número de elétrons), s contínuo (potencial).
Capítulo 8.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def taxas_tunelamento(
    s: np.ndarray,
    q: int,
    Gamma0: float = 1.0,
    alpha: float = 2.0,
    V_bias: float = 0.3,
    E_c: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Taxas de adição (λ^a) e remoção (λ^b) de elétron:

        λ^a(s) ∝ 1 / (1 + exp(α (E_c (q+1/2) - s - V_bias/2)))
        λ^b(s) ∝ 1 / (1 + exp(α (s - E_c (q-1/2) - V_bias/2)))

    Capturam bloqueio de Coulomb e degraus.
    """
    s = np.atleast_1d(s).astype(float)
    la = Gamma0 / (1.0 + np.exp(alpha * (E_c * (q + 0.5) - s - 0.5 * V_bias)))
    lb = Gamma0 / (1.0 + np.exp(alpha * (s - E_c * (q - 0.5) - 0.5 * V_bias)))
    return la, lb


def simular_set(
    n_passos: int = 3000,
    dt: float = 0.01,
    q0: int = 0,
    s0: float = 0.0,
    sigma: float = 0.15,
    Gamma0: float = 1.0,
    E_c: float = 0.5,
    V_bias: float = 0.3,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """
    Simulação híbrida:
      - s: difusão (Langevin)
      - q: saltos ±1 com taxas λ^a, λ^b
    """
    g = np.random.default_rng(semente)
    q = q0
    s = s0
    traj_q = np.zeros(n_passos + 1, dtype=int)
    traj_s = np.zeros(n_passos + 1)
    traj_q[0], traj_s[0] = q, s
    for k in range(n_passos):
        la, lb = taxas_tunelamento(np.array([s]), q, Gamma0, E_c=E_c, V_bias=V_bias)
        # saltos
        if g.random() < float(np.asarray(la).reshape(-1)[0]) * dt:
            q = q + 1
        elif g.random() < float(np.asarray(lb).reshape(-1)[0]) * dt:
            q = max(q - 1, -5)
        # difusão do potencial
        s = s + 0.05 * (V_bias - s) * dt + sigma * g.normal(0, np.sqrt(dt))
        traj_q[k + 1] = q
        traj_s[k + 1] = s
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "q": traj_q, "s": traj_s}
