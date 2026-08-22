"""
Modelo de Dois Fluidos Elétron–Fônon (Two-Temperature Model, TTM).

Não-equilíbrio térmico em canais ~1 nm:
  elétrons aquecem primeiro; a rede (fônons) responde via G_{e-ph}.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def parametros_ttm_default() -> Dict[str, float]:
    """
    Parâmetros efetivos normalizados (unidades SI reduzidas para nanoescala).
    """
    return {
        "C_e": 2.0,       # capacidade calorífica eletrônica
        "C_L": 10.0,       # capacidade da rede (tipicamente > C_e)
        "kappa_e": 0.15,   # condutividade térmica eletrônica
        "kappa_L": 0.08,   # condutividade da rede
        "G": 1.0,         # acoplamento elétron–fônon G_{e-ph}
        "sigma_J": 1.0,   # condutividade elétrica (fonte Joule)
        "E_field": 0.4,   # campo elétrico efetivo |E|
    }


def fonte_joule(E: float, sigma_J: float) -> float:
    """J·E ≈ σ |E|² (aquecimento Joule nos elétrons)."""
    return sigma_J * E ** 2


def passo_ttm_1d(
    Te: np.ndarray,
    TL: np.ndarray,
    dx: float,
    dt: float,
    p: Dict[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integração explícita 1D (diferenças finitas) do TTM:

      C_e ∂_t T_e = κ_e ∂_{xx} T_e − G (T_e − T_L) + σ|E|²
      C_L ∂_t T_L = κ_L ∂_{xx} T_L + G (T_e − T_L)
    """
    Joule = fonte_joule(p["E_field"], p["sigma_J"])
    # laplaciano com BC Neumann (isolamento nas bordas)
    def lap(T):
        L = np.zeros_like(T)
        L[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / (dx ** 2)
        L[0] = (T[1] - T[0]) / (dx ** 2)
        L[-1] = (T[-2] - T[-1]) / (dx ** 2)
        return L

    dTe = (p["kappa_e"] * lap(Te) - p["G"] * (Te - TL) + Joule) / p["C_e"]
    dTL = (p["kappa_L"] * lap(TL) + p["G"] * (Te - TL)) / p["C_L"]
    return Te + dTe * dt, TL + dTL * dt


def simular_ttm(
    n_x: int = 40,
    n_t: int = 800,
    L: float = 1.0,
    t_final: float = 2.0,
    Te0: float = 1.0,
    TL0: float = 1.0,
    hot_spot: float = 0.3,
    p: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray]:
    """Simula TTM 1D com hotspot inicial nos elétrons."""
    if p is None:
        p = parametros_ttm_default()
    x = np.linspace(0, L, n_x)
    dx = x[1] - x[0]
    dt = t_final / n_t
    Te = Te0 + hot_spot * np.exp(-((x - L / 2) / 0.15) ** 2)
    TL = np.full(n_x, TL0)
    traj_e = np.zeros((n_t + 1, n_x))
    traj_L = np.zeros((n_t + 1, n_x))
    traj_e[0], traj_L[0] = Te, TL.copy()
    for k in range(n_t):
        Te, TL = passo_ttm_1d(Te, TL, dx, dt, p)
        Te = np.maximum(Te, 0.1)
        TL = np.maximum(TL, 0.1)
        traj_e[k + 1], traj_L[k + 1] = Te, TL
    t = np.arange(n_t + 1) * dt
    return {"t": t, "x": x, "Te": traj_e, "TL": traj_L, "p": p}
