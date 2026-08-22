"""
Efeito Kondo de Dois Canais (2CK) — não-Fermi líquido.
Entropia residual Majorana S_res = (1/2) k_B ln 2
Condutância: G = G_max [1 − A √(T/T_K)]
"""

from typing import Dict
import numpy as np


def parametros_2ck_default() -> Dict[str, float]:
    return {
        "T_K": 1.0,          # temperatura Kondo
        "G_max": 0.5,        # e²/(2h) em unidades e²/h = 1 → 0.5
        "beta_2CK": 0.8,     # amplitude NFL
        "J1": 1.0,
        "J2": 1.0,           # J1 = J2 (ponto crítico)
        "hbar": 1.0,
        "kB": 1.0,
        "S_res": 0.5 * np.log(2),  # (1/2) ln 2
    }


def G_2CK(V: np.ndarray, T: np.ndarray, p: Dict) -> np.ndarray:
    """
    G(V,T) = G_max [ 1 − β √( max(|eV|, kT) / (k T_K) ) ]
    forma regularizada √(√((eV)²+(kT)²) / kT_K)
    """
    escala = np.sqrt(np.sqrt(V ** 2 + (p["kB"] * T) ** 2) / (p["kB"] * p["T_K"]) + 1e-12)
    G = p["G_max"] * (1.0 - p["beta_2CK"] * escala)
    return np.clip(G, 0.0, p["G_max"])


def G_2CK_T(T: np.ndarray, p: Dict) -> np.ndarray:
    """G(T) em V=0: 1 − A √(T/T_K)"""
    return np.clip(
        p["G_max"] * (1.0 - p["beta_2CK"] * np.sqrt(T / p["T_K"] + 1e-12)),
        0.0,
        p["G_max"],
    )


def entropia_residual(p: Dict) -> float:
    return float(p["S_res"])
