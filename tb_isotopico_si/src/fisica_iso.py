"""
Tight-Binding com desordem isotópica Si²⁸ / Si²⁹ / Si³⁰.
Perturbação de ponto zero δϵ_i quebra degenerescência residual (tipo 1s T₂).
"""

from typing import Dict, Tuple, List
import numpy as np


# massas atômicas (u)
MASSAS = {
    28: 27.9769,
    29: 28.9765,
    30: 29.9738,
}
# abundâncias naturais aproximadas
ABUND = {28: 0.922, 29: 0.047, 30: 0.031}
M_BAR = sum(ABUND[k] * MASSAS[k] for k in MASSAS)


def parametros_iso_default() -> Dict:
    return {
        "n_sites": 21,
        "t_hop": 1.0,
        "E0": 0.0,
        "V_P": -2.0,          # doador no centro
        "alpha_iso": 0.8,     # força da desordem isotópica
        "n_realizacoes": 40,
        "n_estados": 6,
    }


def amostrar_massas(n: int, semente: int = None) -> np.ndarray:
    """Amostra massas com abundâncias naturais."""
    g = np.random.default_rng(semente)
    iso = g.choice([28, 29, 30], size=n, p=[ABUND[28], ABUND[29], ABUND[30]])
    return np.array([MASSAS[i] for i in iso])


def delta_epsilon(massas: np.ndarray, alpha_iso: float = 0.8) -> np.ndarray:
    """δϵ_i = α_iso (M_i − M̄)/M̄"""
    return alpha_iso * (massas - M_BAR) / M_BAR


def hamiltoniano_iso(
    n: int,
    delta_eps: np.ndarray,
    t_hop: float = 1.0,
    E0: float = 0.0,
    V_P: float = -2.0,
) -> np.ndarray:
    """Cadeia 1D TB + desordem onsite + V_P no centro."""
    H = np.zeros((n, n), dtype=float)
    for i in range(n):
        H[i, i] = E0 + delta_eps[i]
        if i < n - 1:
            H[i, i + 1] = -t_hop
            H[i + 1, i] = -t_hop
    H[n // 2, n // 2] += V_P
    return H


def espectro_ensemble(
    p: Dict,
    semente: int = 42,
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """
    Gera n_realizacoes, diagonaliza, retorna:
      evals (n_real, n_estados), lista de H, lista de delta_eps
    """
    g = np.random.default_rng(semente)
    n = p["n_sites"]
    n_r = p["n_realizacoes"]
    n_e = p["n_estados"]
    evals_all = np.zeros((n_r, n_e))
    Hs, deltas = [], []
    for r in range(n_r):
        massas = amostrar_massas(n, semente=int(g.integers(0, 1e9)))
        de = delta_epsilon(massas, p["alpha_iso"])
        H = hamiltoniano_iso(n, de, p["t_hop"], p["E0"], p["V_P"])
        w = np.linalg.eigvalsh(H)[:n_e]
        evals_all[r] = w
        Hs.append(H)
        deltas.append(de)
    return evals_all, Hs, deltas


def splitting_T2_like(evals: np.ndarray) -> np.ndarray:
    """
    Proxy do splitting de estados quase-degenerados:
    diferença entre 2º e 3º autovalores (acima do fundamental).
    """
    if evals.ndim == 1:
        return np.array([evals[2] - evals[1]])
    return evals[:, 2] - evals[:, 1]
