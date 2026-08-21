"""
Resíduo da equação de Poisson 3D e perda composta PINN.
Cap. 2.5 do livro — Formulação da Função de Perda Composta.
"""

import numpy as np
from typing import Callable, Tuple
from .rede_pinn3d import RedePINN3D


def residuo_poisson(
    rede: RedePINN3D,
    X: np.ndarray,
    epsilon_fn: Callable,
    rho_fn: Callable,
) -> np.ndarray:
    """
    Resíduo aproximado: ε Δφ + ρ
    (ε constante por região — válido no interior de cada material).
    Equação alvo: ∇ · (ε ∇φ) = −ρ.
    """
    eps = epsilon_fn(X[:, 0], X[:, 1], X[:, 2])
    lap = rede.laplaciano(X)
    rho = rho_fn(X[:, 0], X[:, 1], X[:, 2])
    return eps * lap + rho


def perda_contorno(
    rede: RedePINN3D,
    X_bc: np.ndarray,
    valores_bc: np.ndarray,
) -> float:
    pred = rede.prever(X_bc)
    return float(np.mean((pred - valores_bc) ** 2))


def perda_composta(
    rede: RedePINN3D,
    X_col: np.ndarray,
    X_bc: np.ndarray,
    valores_bc: np.ndarray,
    epsilon_fn: Callable,
    rho_fn: Callable,
    peso_pde: float = 1.0,
    peso_bc: float = 10.0,
) -> Tuple[float, float, float]:
    """
    J(θ) = peso_pde * MSE(resíduo) + peso_bc * MSE(contorno)
    (Cap. 2.5 — perda composta).
    """
    res = residuo_poisson(rede, X_col, epsilon_fn, rho_fn)
    perda_pde = float(np.mean(res ** 2))
    perda_bc = perda_contorno(rede, X_bc, valores_bc)
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, perda_pde, perda_bc
