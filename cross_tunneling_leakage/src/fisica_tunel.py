"""
Tunelamento quântico direto inter-fio (cross-tunneling leakage).
Dois nanofios vizinhos sob dielétrico sub-1 nm.
"""

from typing import Dict
import numpy as np


def parametros_tunel_default() -> Dict[str, float]:
    """
    Parâmetros efetivos normalizados.
    A barreira Φ_B e d_int entram no pré-fator e no expoente de Fowler–Nordheim / WKB.
    """
    return {
        "d_int": 0.8,       # distância inter-fio (nm, unidades reduzidas)
        "Phi_B": 1.5,       # altura de barreira
        "m_star": 1.0,
        "C1": 1.0,          # capacitância linha 1
        "C2": 1.0,
        "L1": 0.1,          # indutância (pode ser ~0 em DC)
        "L2": 0.1,
        "R1": 0.05,         # resistência série
        "R2": 0.05,
        "G_leak0": 0.3,     # pré-fator efetivo de J_leak / (V1-V2)
    }


def G_tunel_wkb(d_int: float, Phi_B: float, m_star: float = 1.0) -> float:
    """
    Condutância de tunelamento efetiva ~ pré-fator × exp(−κ d_int √Φ_B).
    Forma simplificada da fórmula do usuário.
    """
    kappa = 4.0  # 4π/h √(2m*) efetivo
    return np.exp(-kappa * d_int * np.sqrt(Phi_B))


def J_leak(V1, V2, p: Dict[str, float]):
    """J_leak = G_eff (V1 − V2), com G_eff incluindo o fator WKB."""
    G = p["G_leak0"] * G_tunel_wkb(p["d_int"], p["Phi_B"], p["m_star"])
    return G * (V1 - V2)
