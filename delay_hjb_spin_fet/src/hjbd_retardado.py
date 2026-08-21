"""
Hamilton–Jacobi–Bellman com retardo (espaço estendido).

    V(x, y, t)  com  x = M_t,  y = M_{t−τ}

Capítulo 37.
"""

import numpy as np
from typing import Callable, Tuple


def hamiltoniano_stt(
    x: np.ndarray,
    y: np.ndarray,
    Vx: np.ndarray,
    gamma: float = 0.8,
    beta_mem: float = 0.35,
    alpha_stt: float = 1.0,
    sigma: float = 0.08,
    lambda_u: float = 0.15,
    alvo: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hamiltoniano minimizado em u (controle STT):

        H = min_u {  f(x,y,u)·V_x + (1/2)σ² V_xx + L(x,u) }

    com f = −γ x + α u − β y
         L = (x − alvo)² + λ_u u²

    Controle ótimo (se V_xx não entra em u):
        u* = −(α / (2 λ_u)) V_x
    """
    # custo de estado
    L0 = (x - alvo) ** 2
    # u ótimo
    u_star = - (alpha_stt / (2.0 * lambda_u + 1e-12)) * Vx
    u_star = np.clip(u_star, -2.0, 2.0)
    # drift sob u*
    f = -gamma * x + alpha_stt * u_star - beta_mem * y
    # Hamiltoniano (sem termo V_t)
    H = f * Vx + L0 + lambda_u * u_star ** 2
    return H, u_star


def residuo_delay_hjb(
    V: np.ndarray,
    Vt: np.ndarray,
    Vx: np.ndarray,
    Vxx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    gamma: float = 0.8,
    beta_mem: float = 0.35,
    alpha_stt: float = 1.0,
    sigma: float = 0.08,
    lambda_u: float = 0.15,
    alvo: float = 0.7,
) -> np.ndarray:
    """
    Resíduo da EDP de Bellman no espaço estendido:

        −V_t + H(x, y, V_x, V_xx) ≈ 0

    (convenção de tempo para frente / custo acumulado).
    """
    H, _ = hamiltoniano_stt(
        x, y, Vx, gamma, beta_mem, alpha_stt, sigma, lambda_u, alvo
    )
    # difusão
    H = H + 0.5 * sigma ** 2 * Vxx
    return -Vt + H
