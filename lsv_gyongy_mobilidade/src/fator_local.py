"""
Fator de volatilidade/mobilidade local L(E, x).
Mapeia saturação de velocidade ao campo longitudinal.
"""

import numpy as np
from typing import Optional


def mobilidade_efetiva_dupire(
    E: np.ndarray,
    mu0: float = 1.0,
    E_sat: float = 1.5,
    beta: float = 2.0,
) -> np.ndarray:
    """
    Mobilidade efetiva com saturação de velocidade (modelo Caughey–Thomas):
        μ_eff(E) = μ0 / (1 + (E/E_sat)^β)^{1/β}
    """
    E = np.atleast_1d(E).astype(float)
    return mu0 / (1.0 + (np.abs(E) / E_sat) ** beta) ** (1.0 / beta)


def fator_local_L(
    E: np.ndarray,
    t: float = 0.0,
    mu0: float = 1.0,
    E_sat: float = 1.5,
    beta: float = 2.0,
    E_nu: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Fator local L(E, t) calibrado via condição de Gyöngy:

        L²(E,t) · E[ν_t | E_t = E] = μ_eff²(E, t)

    Se E[ν|E] ≈ E_nu (média condicional estimada), então
        L(E) = μ_eff(E) / sqrt(E[ν|E])
    """
    mu = mobilidade_efetiva_dupire(E, mu0, E_sat, beta)
    if E_nu is None:
        E_nu = np.ones_like(mu)
    return mu / np.sqrt(np.maximum(E_nu, 1e-8))


def velocidade_saturacao(
    E: np.ndarray,
    v_sat: float = 1.0,
    E_sat: float = 1.5,
) -> np.ndarray:
    """v(E) = μ0 E / (1 + E/E_sat) ≈ v_sat · E/(E+E_sat)."""
    E = np.atleast_1d(np.abs(E))
    return v_sat * E / (E + E_sat)
