"""
Condição de Gyöngy e calibração do fator local L.

    L²(E, t) · E[ν_t | E_t = E] = μ_eff²(E, t)

Resolvida numericamente (histograma condicional) e via PINN simples.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from .fator_local import mobilidade_efetiva_dupire, fator_local_L
from .rede_pinn_gyongy import RedePINN1D


def estimativa_condicional_nu(
    E_traj: np.ndarray,
    nu_traj: np.ndarray,
    grade_E: np.ndarray,
    largura: float = 0.15,
) -> np.ndarray:
    """
    Estimativa kernel de E[ν | E = e] ao longo de grade_E.
    """
    media = np.zeros_like(grade_E)
    for i, e in enumerate(grade_E):
        w = np.exp(-0.5 * ((E_traj - e) / largura) ** 2)
        w_sum = np.sum(w)
        if w_sum < 1e-12:
            media[i] = np.mean(nu_traj)
        else:
            media[i] = np.sum(w * nu_traj) / w_sum
    return media


def calibrar_L_gyongy(
    E_traj: np.ndarray,
    nu_traj: np.ndarray,
    grade_E: np.ndarray,
    mu0: float = 1.0,
    E_sat: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Retorna (grade_E, E[ν|E], L(E)) pela condição de Gyöngy.
    """
    E_nu = estimativa_condicional_nu(E_traj, nu_traj, grade_E)
    mu = mobilidade_efetiva_dupire(grade_E, mu0, E_sat)
    L = mu / np.sqrt(np.maximum(E_nu, 1e-8))
    return grade_E, E_nu, L


def perda_gyongy_pinn(
    rede: RedePINN1D,
    grade_E: np.ndarray,
    E_nu: np.ndarray,
    mu0: float = 1.0,
    E_sat: float = 1.5,
) -> float:
    """
    Perda: (L_θ(E)² · E[ν|E] − μ_eff²(E))²
    """
    L = rede.prever(grade_E)
    mu = mobilidade_efetiva_dupire(grade_E, mu0, E_sat)
    residuo = L ** 2 * E_nu - mu ** 2
    return float(np.mean(residuo ** 2))
