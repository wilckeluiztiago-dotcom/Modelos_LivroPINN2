"""
Resíduo da equação de Wigner–Boltzmann (forma semiclassica + dissipação).
"""

import numpy as np
from typing import Tuple
from .celula_dgm import RedeDGM
from .wigner_boltzmann import NanofolhaWigner


def residuo_wigner_reduzido(
    rede: RedeDGM,
    X: np.ndarray,
    nano: NanofolhaWigner,
) -> np.ndarray:
    """
    X: (N, 3) = (x, kx, t)

    Equação de Boltzmann/Wigner semiclassica com relaxação:

        ∂f/∂t + (ℏ k / m) ∂f/∂x + F(x) ∂f/∂k  = −γ (f − f_eq)

    Resíduo = LHS − RHS  (buscamos ≈ 0).
    f_eq ≈ 0 na demonstração (vácuo / pouca ocupação).
    """
    grad = rede.gradiente(X)          # (N, 3) → ∂/∂x, ∂/∂kx, ∂/∂t
    df_dx = grad[:, 0]
    df_dk = grad[:, 1]
    df_dt = grad[:, 2]

    x = X[:, 0]
    kx = X[:, 1]
    v = (nano.hbar * kx) / nano.m_eff   # velocidade de grupo
    F = nano.forca(x)

    # colisão BGK simples
    f = rede.prever(X)
    colisao = -nano.gamma_scatt * f

    return df_dt + v * df_dx + F * df_dk - colisao


def perda_inicial(
    rede: RedeDGM,
    X0: np.ndarray,
    f0: np.ndarray,
) -> float:
    """MSE na condição inicial t=0."""
    pred = rede.prever(X0)
    return float(np.mean((pred - f0) ** 2))


def perda_composta_wigner(
    rede: RedeDGM,
    X_col: np.ndarray,
    X0: np.ndarray,
    f0: np.ndarray,
    nano: NanofolhaWigner,
    peso_pde: float = 1.0,
    peso_ic: float = 10.0,
) -> Tuple[float, float, float]:
    res = residuo_wigner_reduzido(rede, X_col, nano)
    perda_pde = float(np.mean(res ** 2))
    perda_ic = perda_inicial(rede, X0, f0)
    total = peso_pde * perda_pde + peso_ic * perda_ic
    return total, perda_pde, perda_ic
