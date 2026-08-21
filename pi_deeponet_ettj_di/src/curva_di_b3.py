"""
Curva de juros DI Futuro (B3) — convenção 252 dias úteis, capitalização composta.
"""

import numpy as np
from typing import Optional, Dict, Tuple


# Vértices líquidos típicos (anos aproximados)
VERTICES_DI = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0])


def taxa_composta_252(r_aa: float, tau: float) -> float:
    """
    Fator de desconto B3:
        DF = 1 / (1 + r)^{tau}
    com tau = DU/252 (anos de dias úteis).
    """
    return 1.0 / ((1.0 + r_aa) ** tau)


def gerar_curva_di(
    vertices: np.ndarray = None,
    r0: float = 0.12,
    slope: float = 0.01,
    curvatura: float = -0.002,
    ruido: float = 0.001,
    semente: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Gera curva sintética de taxas DI (a.a.) nos vértices.
    r(τ) ≈ r0 + slope·τ + curvatura·τ² + ruído
    """
    if vertices is None:
        vertices = VERTICES_DI.copy()
    g = np.random.default_rng(semente)
    r = r0 + slope * vertices + curvatura * vertices ** 2
    r = r + g.normal(0, ruido, size=len(vertices))
    r = np.clip(r, 0.01, 0.40)
    return vertices, r


def forward_instantanea(
    vertices: np.ndarray,
    taxas: np.ndarray,
    T_grid: np.ndarray,
) -> np.ndarray:
    """
    Forward f(0,T) aproximada por interpolação log-linear dos DFs.
    """
    # DF nos vértices
    df = np.array([taxa_composta_252(r, t) for t, r in zip(vertices, taxas)])
    # interpola log DF
    log_df = np.interp(T_grid, vertices, np.log(np.maximum(df, 1e-12)))
    # f ≈ -∂ log DF / ∂T
    dT = np.gradient(T_grid)
    f = -np.gradient(log_df) / (dT + 1e-15)
    return np.maximum(f, 0.0)


def preco_titulo_df(
    vertices: np.ndarray,
    taxas: np.ndarray,
    t: float,
    T: float,
) -> float:
    """
    P(t,T) ≈ DF(T)/DF(t) sob curva estática (marcação simplificada).
    """
    if T <= t:
        return 1.0
    # interpola taxa
    r_T = float(np.interp(T, vertices, taxas))
    r_t = float(np.interp(max(t, vertices[0]), vertices, taxas))
    df_T = taxa_composta_252(r_T, T)
    df_t = taxa_composta_252(r_t, max(t, 1e-6))
    return float(df_T / df_t)


def gerar_superficie_P(
    vertices: np.ndarray,
    taxas: np.ndarray,
    t_grid: np.ndarray,
    T_grid: np.ndarray,
) -> np.ndarray:
    """Matriz P[i,j] = P(t_i, T_j)."""
    P = np.zeros((len(t_grid), len(T_grid)))
    for i, t in enumerate(t_grid):
        for j, T in enumerate(T_grid):
            P[i, j] = preco_titulo_df(vertices, taxas, t, T) if T >= t else 1.0
    return P
