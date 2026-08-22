"""
Banho nuclear ²⁹Si e descoerência T₂* do ³¹P.

Concentração natural ~4.7% de ²⁹Si (I=1/2).
Acoplamento dipolar + flip-flop do banho.
"""

from typing import Dict, Tuple
import numpy as np


def parametros_banho_default() -> Dict[str, float]:
    return {
        "c_29Si": 0.047,     # fração isotópica
        "a_Si": 0.543,       # parâmetro de rede Si (nm)
        "gamma_P": 1.0,      # γ_P normalizado
        "gamma_29": 0.5,     # γ_29 / γ_P efetivo
        "mu0_hbar": 1.0,     # μ0 ħ / (4π) efetivo
        "R_max": 5.0,        # raio de cutoff do banho (nm u.a.)
        "N_bath": 80,        # número de spins ²⁹Si amostrados
    }


def gerar_banho_29Si(
    n: int = 80,
    R_max: float = 5.0,
    semente: int = 42,
) -> np.ndarray:
    """
    Posições aleatórias de ²⁹Si em esfera de raio R_max
    (amostra efetiva do banho diluído).
    Retorna array (n, 3).
    """
    g = np.random.default_rng(semente)
    # distribuição uniforme em volume
    u = g.random(n)
    r = R_max * u ** (1.0 / 3.0)
    # evitar r muito pequeno (próximo do ³¹P)
    r = np.maximum(r, 0.3)
    theta = np.arccos(2 * g.random(n) - 1)
    phi = 2 * np.pi * g.random(n)
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.column_stack([x, y, z])


def acoplamentos_dipolares(
    pos: np.ndarray,
    gamma_P: float = 1.0,
    gamma_29: float = 0.5,
    mu0_hbar: float = 1.0,
) -> np.ndarray:
    """
    A_k^dipolar ∝ (γ_P γ_29 / r_k³) (1 − 3 cos² θ_k)
    (termo secular I_P^z I_k^z efetivo para dephasing).
    """
    r = np.linalg.norm(pos, axis=1)
    cos_th = pos[:, 2] / np.maximum(r, 1e-8)
    geom = 1.0 - 3.0 * cos_th ** 2
    A = mu0_hbar * gamma_P * gamma_29 / (r ** 3) * geom
    return A


def T2_star_de_A(A: np.ndarray) -> float:
    """
    1/(T₂*)² = (1/2) Σ_k |A_k|²
    → T₂* = √2 / √(Σ A_k²)
    """
    s = np.sum(A ** 2)
    if s < 1e-30:
        return 1e6
    return float(np.sqrt(2.0 / s))


def fid_gaussiano(t: np.ndarray, T2s: float) -> np.ndarray:
    """⟨S_x(t)⟩ = exp(−(t/T₂*)²)"""
    return np.exp(-(t / max(T2s, 1e-12)) ** 2)
