"""
Equação mestre para P(N,t) no SET:

  ∂P(N)/∂t = Σ_{N'} [ Γ(N'→N) P(N') − Γ(N→N') P(N) ]

Apenas saltos N↔N±1 (tunelamento de um elétron).
"""

import numpy as np
from typing import Optional, Dict
from .fisica_set import taxas_tunelamento


def passo_mestre(
    P: np.ndarray,
    N_vals: np.ndarray,
    dt: float,
    V_g: float,
    V_sd: float,
    **kwargs,
) -> np.ndarray:
    """Integração de Euler da equação mestre (conserva ΣP ≈ 1)."""
    dP = np.zeros_like(P)
    n = len(N_vals)
    for i, N in enumerate(N_vals):
        ga, gr = taxas_tunelamento(int(N), V_g, V_sd, **kwargs)
        # saída
        dP[i] -= (ga + gr) * P[i]
        # entrada de N-1 via adição
        if i > 0:
            ga_m, _ = taxas_tunelamento(int(N_vals[i - 1]), V_g, V_sd, **kwargs)
            dP[i] += ga_m * P[i - 1]
        # entrada de N+1 via remoção
        if i < n - 1:
            _, gr_p = taxas_tunelamento(int(N_vals[i + 1]), V_g, V_sd, **kwargs)
            dP[i] += gr_p * P[i + 1]
    P_new = P + dP * dt
    P_new = np.maximum(P_new, 0.0)
    s = P_new.sum()
    if s > 1e-15:
        P_new /= s
    return P_new


def simular_mestre(
    N_min: int = -2,
    N_max: int = 5,
    n_passos: int = 500,
    dt: float = 0.02,
    V_g: float = 1.0,
    V_sd: float = 0.3,
    P0: Optional[np.ndarray] = None,
    semente: Optional[int] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    N_vals = np.arange(N_min, N_max + 1)
    if P0 is None:
        P = np.zeros(len(N_vals))
        P[np.argmin(np.abs(N_vals - 0))] = 1.0
    else:
        P = P0.copy()
    traj = np.zeros((n_passos + 1, len(N_vals)))
    traj[0] = P
    for k in range(n_passos):
        P = passo_mestre(P, N_vals, dt, V_g, V_sd, **kwargs)
        traj[k + 1] = P
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "N_vals": N_vals, "P": traj}


def varredura_gate(
    V_g_vals: np.ndarray,
    N_min: int = -2,
    N_max: int = 5,
    V_sd: float = 0.25,
    n_relax: int = 400,
    dt: float = 0.02,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Varredura de V_g → corrente média estacionária (degraus de Coulomb)."""
    from .fisica_set import corrente_media
    N_vals = np.arange(N_min, N_max + 1)
    I = np.zeros(len(V_g_vals))
    P_stat = np.zeros((len(V_g_vals), len(N_vals)))
    for j, Vg in enumerate(V_g_vals):
        P = np.zeros(len(N_vals))
        P[len(N_vals) // 2] = 1.0
        for _ in range(n_relax):
            P = passo_mestre(P, N_vals, dt, Vg, V_sd, **kwargs)
        P_stat[j] = P
        I[j] = corrente_media(P, N_vals, Vg, V_sd, **kwargs)
    return {"V_g": V_g_vals, "I": I, "P_stat": P_stat, "N_vals": N_vals}
