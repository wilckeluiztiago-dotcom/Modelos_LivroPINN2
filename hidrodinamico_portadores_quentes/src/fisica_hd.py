"""
Modelo hidrodinâmico de portadores quentes (Baccarani–Wordeman / Bløtekjaer).
Velocity overshoot sob campos altos em canais sub-10 nm.
"""

from typing import Dict
import numpy as np


def parametros_hd_default() -> Dict[str, float]:
    return {
        "q": 1.0,
        "m_star": 1.0,
        "kB": 1.0,
        "T0": 1.0,           # temperatura da rede
        "tau_p0": 0.5,       # tempo de relaxação de momento
        "tau_w0": 1.0,       # tempo de relaxação de energia
        "kappa_n": 0.1,      # condutividade térmica eletrônica
        "E_field": 1.2,      # campo elétrico médio (overshoot)
        "n0": 1.0,           # densidade de referência
        "L": 1.0,
    }


def tau_p(Tn: np.ndarray, p: Dict) -> np.ndarray:
    """τ_p(T_n) — diminui com T_n (espalhamento mais forte)."""
    return p["tau_p0"] * np.sqrt(p["T0"] / np.maximum(Tn, 0.1))


def tau_w(Tn: np.ndarray, p: Dict) -> np.ndarray:
    return p["tau_w0"] * (p["T0"] / np.maximum(Tn, 0.1))


def W_n(Tn: np.ndarray, vn: np.ndarray, p: Dict) -> np.ndarray:
    """W_n = 3/2 kT_n + 1/2 m* v²"""
    return 1.5 * p["kB"] * Tn + 0.5 * p["m_star"] * vn ** 2


def W0(p: Dict) -> float:
    return 1.5 * p["kB"] * p["T0"]


def perfil_campo_1d(x: np.ndarray, p: Dict) -> np.ndarray:
    """Campo com pico no canal (overshoot region)."""
    L = p["L"]
    E0 = p["E_field"]
    return E0 * (1.0 + 0.5 * np.exp(-((x - 0.5 * L) / 0.15) ** 2))
