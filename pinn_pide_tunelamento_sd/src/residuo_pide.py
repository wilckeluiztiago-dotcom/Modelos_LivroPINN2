"""
Resíduo da equação integro-diferencial (PIDE) de continuidade + tunelamento.

∂n/∂t + ∂J/∂x = G_tun[n]

onde G_tun é o operador integral de penetração de barreira amostrado
por Monte Carlo contínuo.
"""

import numpy as np
from typing import Tuple, Optional
from .rede_pinn import RedePINN
from .barreira_tunelamento import CanalSub12nm, kernel_tunelamento


def corrente_drift_diffusion(
    rede: RedePINN,
    x: np.ndarray,
    mu: float = 0.8,
    D: float = 0.05,
    campo_E: Optional[np.ndarray] = None,
) -> np.ndarray:
    """J = μ n E − D ∂n/∂x  (estacionário: usamos n = φ_θ)."""
    n = rede.prever(x)
    dn = rede.gradiente(x)
    if campo_E is None:
        campo_E = np.ones_like(x) * 0.5
    return mu * n * campo_E - D * dn


def operador_tunelamento_mc(
    rede: RedePINN,
    x: np.ndarray,
    canal: CanalSub12nm,
    n_mc: int = 32,
    semente: Optional[int] = None,
) -> np.ndarray:
    """
    Operador integral amostrado por Monte Carlo contínuo:

        G_tun[n](x) ≈ (L / N_mc) Σ_j K(x, y_j) (n(y_j) − n(x))

    O kernel K codifica a penetração de barreira (WKB).
    Acoplado ao operador diferencial de continuidade.
    """
    g = np.random.default_rng(semente)
    y = g.uniform(0.0, canal.L, size=n_mc)
    n_x = rede.prever(x)
    n_y = rede.prever(y)

    G = np.zeros_like(x, dtype=float)
    for i, xi in enumerate(np.atleast_1d(x)):
        K = kernel_tunelamento(
            np.full(n_mc, xi), y, canal, E_ref=0.12, alpha=2.5
        )
        # ganho − perda local
        G[i] = canal.L * np.mean(K * (n_y - n_x[i]))
    return G


def residuo_pide_estacionario(
    rede: RedePINN,
    x: np.ndarray,
    canal: CanalSub12nm,
    mu: float = 0.8,
    D: float = 0.05,
    n_mc: int = 24,
    semente: Optional[int] = None,
) -> np.ndarray:
    """
    Resíduo estacionário da PIDE:

        dJ/dx − G_tun[n]  ≈ 0

    (continuidade: ∂n/∂t = 0 = −∂J/∂x + G_tun).
    """
    # ∂J/∂x por diferenças
    eps = 1e-4
    Jp = corrente_drift_diffusion(rede, x + eps, mu, D)
    Jm = corrente_drift_diffusion(rede, x - eps, mu, D)
    dJ = (Jp - Jm) / (2 * eps)
    G = operador_tunelamento_mc(rede, x, canal, n_mc=n_mc, semente=semente)
    return dJ - G


def perda_pide(
    rede: RedePINN,
    x_col: np.ndarray,
    x_bc: np.ndarray,
    n_bc: np.ndarray,
    canal: CanalSub12nm,
    peso_pde: float = 1.0,
    peso_bc: float = 12.0,
    n_mc: int = 20,
    semente: Optional[int] = None,
) -> Tuple[float, float, float]:
    res = residuo_pide_estacionario(rede, x_col, canal, n_mc=n_mc, semente=semente)
    perda_pde = float(np.mean(res ** 2))
    pred_bc = rede.prever(x_bc)
    perda_bc = float(np.mean((pred_bc - n_bc) ** 2))
    return peso_pde * perda_pde + peso_bc * perda_bc, perda_pde, perda_bc
