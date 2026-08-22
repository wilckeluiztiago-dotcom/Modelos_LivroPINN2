"""
Diodo de Tunelamento Ressonante de Dupla Barreira (DBRTD).
Formalismo de Tsu–Esaki + transmissão por matriz de transferência.
"""

from typing import Dict, Tuple
import numpy as np


def parametros_dbrtd_default() -> Dict:
    return {
        "hbar2_2m": 0.5,     # ħ²/(2m*) unidades
        "m_star": 1.0,
        "L_total": 1.0,
        "x_b1": 0.25,        # início barreira 1
        "w_b": 0.12,         # largura de cada barreira
        "w_w": 0.20,         # largura do poço
        "V0": 5.0,           # altura das barreiras
        "E_F": 1.5,
        "kT": 0.08,
        "n_x": 400,
    }


def potencial_dupla_barreira(
    x: np.ndarray,
    V_bias: float = 0.0,
    p: Dict = None,
) -> np.ndarray:
    """
    Duas barreiras + poço central; bias linear ao longo do dispositivo.
    """
    if p is None:
        p = parametros_dbrtd_default()
    V = np.zeros_like(x)
    xb1 = p["x_b1"]
    wb = p["w_b"]
    ww = p["w_w"]
    xb2 = xb1 + wb + ww
    V0 = p["V0"]
    # barreiras
    V[(x >= xb1) & (x < xb1 + wb)] = V0
    V[(x >= xb2) & (x < xb2 + wb)] = V0
    # queda linear de bias
    V = V - V_bias * (x / max(p["L_total"], 1e-12))
    return V


def transmissao_transfer_matrix(
    E: float,
    V_bias: float = 0.0,
    p: Dict = None,
    n_pts: int = 500,
) -> float:
    """
    Matriz de transferência 1D para T(E).
    Regiões de potencial constante por fatia.
    """
    if p is None:
        p = parametros_dbrtd_default()
    if E <= 1e-8:
        return 0.0
    L = p["L_total"]
    x = np.linspace(0, L, n_pts)
    dx = x[1] - x[0]
    V = potencial_dupla_barreira(x, V_bias, p)
    h2m = p["hbar2_2m"]

    # k em cada fatia (complexo se E < V)
    k = np.sqrt((E - V) / h2m + 0j)
    # evitar k=0
    k = np.where(np.abs(k) < 1e-10, 1e-10 + 0j, k)

    # matriz de transferência acumulada (da esquerda para direita)
    # interface + propagação
    M = np.eye(2, dtype=complex)
    for i in range(len(x) - 1):
        k1, k2 = k[i], k[i + 1]
        # interface contínua ψ, ψ'
        if abs(k1) > 1e-14:
            T_int = 0.5 * np.array([
                [1 + k2 / k1, 1 - k2 / k1],
                [1 - k2 / k1, 1 + k2 / k1],
            ], dtype=complex)
        else:
            T_int = np.eye(2, dtype=complex)
        # propagação em fatia i+1
        phase = np.exp(1j * k2 * dx)
        T_prop = np.array([
            [phase, 0],
            [0, 1.0 / phase if abs(phase) > 1e-30 else 1.0],
        ], dtype=complex)
        # na verdade ordem: propaga em região i depois interface
        phase1 = np.exp(1j * k1 * dx)
        T_p = np.diag([phase1, 1.0 / phase1 if abs(phase1) > 1e-30 else 1.0])
        M = T_int @ T_p @ M

    # T = |1/M_22|² * (kR/kL) para corrente
    kL = np.sqrt(E / h2m + 0j)
    kR = np.sqrt(max(E + V_bias, 1e-12) / h2m + 0j)
    M22 = M[1, 1]
    if abs(M22) < 1e-30:
        return 0.0
    T = (np.real(kR) / max(np.real(kL), 1e-12)) * (1.0 / abs(M22) ** 2)
    # clamp
    return float(np.clip(T.real if np.iscomplexobj(T) else T, 0.0, 5.0))


def corrente_tsu_esaki(
    V_bias: float,
    p: Dict = None,
    n_E: int = 120,
) -> float:
    """
    J(V) fórmula de Tsu–Esaki (integral 1D efetiva).
    """
    if p is None:
        p = parametros_dbrtd_default()
    if abs(V_bias) < 1e-8:
        V_bias = 1e-4
    EF, kT = p["E_F"], p["kT"]
    E_max = EF + 8 * kT + abs(V_bias)
    Ex = np.linspace(0.02, E_max, n_E)
    dE = Ex[1] - Ex[0]
    T = np.array([transmissao_transfer_matrix(e, V_bias, p) for e in Ex])
    # fator logarítmico Tsu-Esaki
    num = 1.0 + np.exp((EF - Ex) / kT)
    den = 1.0 + np.exp((EF - Ex - V_bias) / kT)
    log_term = np.log(np.maximum(num / np.maximum(den, 1e-30), 1e-30))
    # pré-fator normalizado = 1
    J = float(np.sum(T * log_term) * dE)
    return max(J, 0.0)


def curva_JV(
    V_vals: np.ndarray,
    p: Dict = None,
) -> np.ndarray:
    return np.array([corrente_tsu_esaki(V, p) for V in V_vals])
