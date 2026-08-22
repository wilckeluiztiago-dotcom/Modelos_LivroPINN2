"""
Ruptura eletrostática em gaps sub-5 nm.
Paschen clássico falha; emissão Fowler–Nordheim domina.
"""

from typing import Dict
import numpy as np


def parametros_fn_default() -> Dict[str, float]:
    return {
        "eps": 1.0,
        "q": 1.0,
        "mu_e": 1.0,        # mobilidade eletrônica
        "D_e": 0.05,        # difusão
        "alpha0": 0.5,      # coeficiente de ionização (Townsend reduzido)
        "A_FN": 2.0,        # pré-fator Fowler–Nordheim
        "B_FN": 3.0,        # expoente FN
        "n_i0": 0.1,        # densidade iônica de fundo (quase-estática)
        "d_gap": 1.0,       # gap normalizado (sub-5 nm)
    }


def G_FN(E: np.ndarray, A_FN: float, B_FN: float) -> np.ndarray:
    """G_FN(E) = A E² exp(−B/|E|)  (emissão de campo)."""
    E_abs = np.maximum(np.abs(E), 1e-6)
    return A_FN * E_abs ** 2 * np.exp(-B_FN / E_abs)


def tensao_paschen_classica(p_d: np.ndarray, A: float = 12.0, B: float = 365.0, gamma: float = 0.01) -> np.ndarray:
    """
    Curva de Paschen clássica: V_b(pd) ≈ B pd / ln(A pd / ln(1+1/γ))
    (apenas referência — inválida em d ≲ 5 nm).
    """
    pd = np.maximum(p_d, 1e-3)
    return B * pd / np.log(np.maximum(A * pd / np.log(1.0 + 1.0 / gamma), 1.01))


def tensao_fn_gap(d: np.ndarray, B_FN: float = 3.0, E_crit: float = 2.0) -> np.ndarray:
    """
    Em escala nm, V_b ~ E_crit * d (emissão de campo),
    monótona decrescente com d — oposta ao braço esquerdo de Paschen.
    """
    return E_crit * d + 0.5 * B_FN * np.sqrt(d)
