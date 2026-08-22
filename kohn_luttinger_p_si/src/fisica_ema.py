"""
Teoria da massa efetiva (EMA) + correção de célula central
para doador ³¹P em silício (Kohn–Luttinger / valley-orbit).

Estados 1s por simetria Td:
  A1 (singlete)  Eb ≈ 45.6 meV
  T2 (triplete)  Eb ≈ 33.9 meV
  E  (dublete)   Eb ≈ 31.3 meV
"""

from typing import Dict
import numpy as np


# energias de ligação experimentais (meV) — referência
E_BIND = {
    "A1": 45.6,
    "T2": 33.9,
    "E": 31.3,
}


def parametros_ema_default() -> Dict[str, float]:
    """
    Unidades efetivas: ħ = 1, m* ≈ 1, energias em meV normalizadas.
    V0, r0 controlam a correção de célula central.
    """
    return {
        "m_star": 1.0,
        "eps_r": 11.7,      # Si
        "V0": 8.0,          # profundidade célula central
        "r0": 0.15,         # alcance (unidades de Bohr efetivo)
        "E_ref": {
            "A1": -45.6 / 45.6,   # normalizado ao A1
            "T2": -33.9 / 45.6,
            "E": -31.3 / 45.6,
        },
    }


def V_coul(r: np.ndarray, eps_r: float = 11.7) -> np.ndarray:
    """V_coul = −1/(ε_r r)  (unidades e²/4πε0 = 1)."""
    return -1.0 / (eps_r * np.maximum(r, 1e-6))


def V_cc(r: np.ndarray, V0: float = 8.0, r0: float = 0.15) -> np.ndarray:
    """Correção de célula central: −V0 exp(−r/r0)."""
    return -V0 * np.exp(-r / r0)


def V_total(r: np.ndarray, p: Dict) -> np.ndarray:
    return V_coul(r, p["eps_r"]) + V_cc(r, p["V0"], p["r0"])
