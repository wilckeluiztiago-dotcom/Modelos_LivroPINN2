"""
Linha de transmissão quântica — CNTs e GNRs.
Indutância cinética L_K e capacitância quântica C_Q dominam
sobre L_mag e C_es clássicos.
"""

from typing import Dict
import numpy as np


def parametros_qtl_default() -> Dict[str, float]:
    """
    Valores efetivos normalizados inspirados em:
      L_K ~ h/(4 e² v_F) ~ 8 nH/μm
      C_Q ~ 4 e²/(h v_F) ~ 100 aF/μm
    """
    L_K = 1.0
    L_mag = 0.05
    C_Q = 1.0
    C_es = 0.2
    C_eff = 1.0 / (1.0 / C_es + 1.0 / C_Q)  # série
    return {
        "R": 0.05,           # R_dist
        "G": 0.01,           # G_dist (perda dielétrica)
        "L_K": L_K,
        "L_mag": L_mag,
        "L_tot": L_mag + L_K,
        "C_Q": C_Q,
        "C_es": C_es,
        "C_eff": C_eff,
        "Z_L": 1.0,          # carga terminadora
        "v_F": 1.0,          # velocidade de Fermi normalizada
    }


def impedancia_caracteristica(p: Dict[str, float]) -> float:
    """Z_0 ≈ √(L_tot / C_eff)"""
    return float(np.sqrt(p["L_tot"] / max(p["C_eff"], 1e-12)))


def velocidade_onda(p: Dict[str, float]) -> float:
    """v ≈ 1/√(L_tot C_eff)  (pode ser ~ v_F em limite quântico)"""
    return float(1.0 / np.sqrt(p["L_tot"] * max(p["C_eff"], 1e-12)))
