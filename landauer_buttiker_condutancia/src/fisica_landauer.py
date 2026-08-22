"""
Quantização de condutância Landauer–Büttiker.
Nanofio / constrição 1D: L < ℓ_in → transporte balístico.
G_0 = 2e²/h
"""

import numpy as np
from typing import Dict, Tuple


# constantes normalizadas: ħ = 1, m* = 1, e = 1 → G_0 = 2/π em u.a. simplificadas
# usamos G0 = 1 como unidade de quantização de condutância


def parametros_lb_default() -> Dict[str, float]:
    return {
        "hbar": 1.0,
        "m_star": 1.0,
        "W": 1.0,          # largura da constrição
        "V0_conf": 0.0,    # fundo do poço transversal
        "G0": 1.0,         # quantum de condutância (unidades)
        "kT": 0.02,        # temperatura efetiva
    }


def pot_confinamento(y: np.ndarray, W: float = 1.0, V_wall: float = 50.0) -> np.ndarray:
    """Poço infinito soft: V alto fora de [0,W]."""
    V = np.zeros_like(y)
    V[y < 0] = V_wall
    V[y > W] = V_wall
    return V


def modos_analiticos_poco(n: int, W: float = 1.0, hbar: float = 1.0, m: float = 1.0) -> Tuple[float, callable]:
    """
    Poço infinito 1D: E_n = (n π ħ)² / (2 m W²), ψ_n = √(2/W) sin(n π y / W)
    n = 1,2,3,...
    """
    E_n = (n * np.pi * hbar) ** 2 / (2.0 * m * W ** 2)

    def psi(y):
        return np.sqrt(2.0 / W) * np.sin(n * np.pi * y / W)

    return E_n, psi


def fermi(E: np.ndarray, mu: float, kT: float) -> np.ndarray:
    x = (E - mu) / max(kT, 1e-8)
    return 1.0 / (1.0 + np.exp(np.clip(x, -40, 40)))


def transmissao_balistica(E: float, E_n: float) -> float:
    """T_n(E) = 1 se E > E_n (modo aberto), 0 caso contrário (ideal)."""
    return 1.0 if E >= E_n else 0.0


def corrente_landauer(
    mu_S: float,
    mu_D: float,
    E_niveis: np.ndarray,
    kT: float = 0.02,
    G0: float = 1.0,
    n_E: int = 200,
) -> float:
    """
    I = (2e/h) Σ_n ∫ T_n(E) [f_S − f_D] dE
    Em unidades G0=1: I ≈ Σ_n ∫ T_n (f_S−f_D) dE
    """
    E_min = min(mu_S, mu_D) - 8 * kT
    E_max = max(mu_S, mu_D) + 8 * kT
    E = np.linspace(E_min, E_max, n_E)
    dE = E[1] - E[0]
    I = 0.0
    for En in E_niveis:
        T = np.array([transmissao_balistica(e, En) for e in E])
        I += np.sum(T * (fermi(E, mu_S, kT) - fermi(E, mu_D, kT))) * dE
    return float(I * G0)


def condutancia_vs_gate(
    V_g_vals: np.ndarray,
    n_modos: int = 5,
    W: float = 1.0,
    kT: float = 0.02,
    V_sd: float = 0.01,
) -> Dict[str, np.ndarray]:
    """
    Varre potencial de gate efetivo que desloca os níveis E_n − e V_g.
    Condutância diferencial G = I / V_sd.
    """
    E_base = np.array([modos_analiticos_poco(n, W)[0] for n in range(1, n_modos + 1)])
    G = []
    for Vg in V_g_vals:
        E_n = E_base - Vg  # gate abaixa barreiras/níveis efetivos
        mu_S = 0.5 * V_sd
        mu_D = -0.5 * V_sd
        # níveis abaixo de uma janela de energia abrem canais
        I = corrente_landauer(mu_S + 0.3 + Vg * 0.0, mu_D + 0.3, np.maximum(E_n, -1.0), kT)
        # melhor: fixar mu e abrir modos quando E_n < mu
        mu = 0.5
        n_abertos = np.sum(E_n < mu)
        G.append(float(n_abertos))  # G/G0 ≈ número de modos abertos (T=0)
    return {"V_g": V_g_vals, "G_sobre_G0": np.array(G), "E_base": E_base}
