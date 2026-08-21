"""
Mapas sintéticos de grãos metálicos (TiN/TaN) e work function (WFF).
Nó 1.6 nm — Capítulo 44.
"""

import numpy as np
from typing import Optional, Tuple


# Work functions típicos por orientação de grão (eV, normalizados)
WF_ORIENTACOES = {
    0: 0.85,   # (100)
    1: 1.00,   # (110)
    2: 1.15,   # (111)
}


def gerar_mapa_graos(
    nx: int = 32,
    ny: int = 16,
    n_graos: int = 12,
    semente: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera mapa de orientação de grãos (Voronoi discreto) e mapa de WF.
    Retorna (orientacao [nx,ny] int, wf [nx,ny] float).
    """
    g = np.random.default_rng(semente)
    # centros de grãos
    cx = g.uniform(0, nx, n_graos)
    cy = g.uniform(0, ny, n_graos)
    orient = g.integers(0, 3, n_graos)

    xx, yy = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    mapa_o = np.zeros((nx, ny), dtype=int)
    for i in range(nx):
        for j in range(ny):
            d2 = (cx - i) ** 2 + (cy - j) ** 2
            mapa_o[i, j] = orient[np.argmin(d2)]

    mapa_wf = np.zeros((nx, ny))
    for k, wf in WF_ORIENTACOES.items():
        mapa_wf[mapa_o == k] = wf
    # ruído fino de interface
    mapa_wf += g.normal(0, 0.02, size=(nx, ny))
    return mapa_o, mapa_wf


def potencial_de_wf(
    wf: np.ndarray,
    V_source: float = 0.0,
    V_drain: float = 0.3,
) -> np.ndarray:
    """
    Potencial eletrostático 1D efetivo ao longo de x (média em y),
    influenciado pelo WF médio local + rampa S/D.
    φ(x) ≈ V_s + (V_d-V_s)(x/L) + α (WF̄(x) - WF_ref)
    """
    nx, ny = wf.shape
    wf_x = wf.mean(axis=1)
    x = np.linspace(0, 1, nx)
    phi = V_source + (V_drain - V_source) * x + 0.4 * (wf_x - 1.0)
    return phi


def residuo_poisson_1d(
    phi: np.ndarray,
    rho: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """‖ε φ'' + ρ‖² aproximado por diferenças finitas."""
    dx = 1.0 / max(len(phi) - 1, 1)
    if len(phi) < 3:
        return 0.0
    lap = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / (dx ** 2)
    r = epsilon * lap + rho[1:-1]
    return float(np.mean(r ** 2))
