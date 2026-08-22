"""
BTBT Kane/Keldysh para TFETs.
Geração de tunelamento interbandas sob campo alto.
"""

from typing import Dict
import numpy as np


def parametros_btbt_default() -> Dict[str, float]:
    return {
        "A_Kane": 2.0,
        "B_Kane": 3.0,
        "E_g": 1.0,          # bandgap normalizado
        "eps": 1.0,
        "q": 1.0,
        "mu_n": 1.0,
        "mu_p": 0.5,
        "D_n": 0.1,
        "D_p": 0.05,
        "n_i": 0.05,         # intrínseco
        "tau_R": 2.0,        # recombinacão SRH efetiva
        "N_D": 2.0,          # doador (fonte/dreno n+)
        "N_A": 2.0,          # aceitador (fonte p+ em TFET)
        "L": 1.0,
    }


def G_Kane(E: np.ndarray, p: Dict) -> np.ndarray:
    """
    G_BTBT = A E² / √E_g  exp(−B E_g^{3/2} / |E|)
    """
    E_abs = np.maximum(np.abs(E), 1e-4)
    Eg = p["E_g"]
    return (
        p["A_Kane"] * E_abs ** 2 / np.sqrt(Eg)
        * np.exp(-p["B_Kane"] * Eg ** 1.5 / E_abs)
    )


def R_srh(n: np.ndarray, p_h: np.ndarray, p: Dict) -> np.ndarray:
    """R ≈ (np − n_i²) / (τ (n+p+2n_i))"""
    ni = p["n_i"]
    return (n * p_h - ni ** 2) / (p["tau_R"] * (n + p_h + 2 * ni) + 1e-8)


def doping_tfet_1d(x: np.ndarray, p: Dict) -> np.ndarray:
    """
    Perfil TFET simplificado: p+ fonte | intrínseco canal | n+ dreno
    N_net = N_D − N_A
    """
    L = p["L"]
    N = np.zeros_like(x)
    N[x < 0.3 * L] = -p["N_A"]      # fonte p+
    N[x > 0.7 * L] = p["N_D"]       # dreno n+
    return N
