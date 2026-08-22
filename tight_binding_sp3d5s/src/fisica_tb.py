"""
Tight-Binding atomístico multi-orbital sp³d⁵s* (10 orbitais/átomo)
com potencial de triagem de Coulomb do doador ³¹P em Si.
"""

from typing import Dict, List, Tuple
import numpy as np


# 10 orbitais: s, px, py, pz, dxy, dyz, dzx, dx2y2, dz2, s*
ORBITAIS = ["s", "px", "py", "pz", "dxy", "dyz", "dzx", "dx2y2", "dz2", "s*"]
N_ORB = 10


def parametros_tb_default() -> Dict:
    """
    Parâmetros efetivos (unidades arbitrárias normalizadas).
    Onsite e hoppings simplificados estilo Jancu/Boykin.
    """
    # energias onsite (Si)
    E_onsite = {
        "s": -2.0,
        "px": 1.0, "py": 1.0, "pz": 1.0,
        "dxy": 3.0, "dyz": 3.0, "dzx": 3.0, "dx2y2": 3.0, "dz2": 3.0,
        "s*": 5.0,
    }
    # hoppings NN (escalares efetivos por tipo)
    hop = {
        "ss": -1.5,
        "sp": 1.2,
        "pp_sigma": 2.0,
        "pp_pi": -0.5,
        "sd": 0.8,
        "s*s": -0.4,
        "s*p": 0.6,
    }
    return {
        "E_onsite": E_onsite,
        "hop": hop,
        "a": 1.0,            # parâmetro de rede (u.a.)
        "eps_r": 11.7,
        "r_core": 0.25,
        "U_cc": -4.0,        # correção de célula central no sítio P
        "n_shells": 2,       # camadas de vizinhos em torno de P
    }


def gerar_cluster_diamante(n_shells: int = 2, a: float = 1.0) -> Tuple[np.ndarray, int]:
    """
    Cluster FCC/diamante centrado no doador (índice 0 = P).
    Gera posições aproximadas de Si em shells cúbicas.
    """
    # vizinhos tetraédricos + shells
    nn = np.array([
        [1, 1, 1],
        [1, -1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
    ], dtype=float) * (a * np.sqrt(3) / 4)
    pos = [np.zeros(3)]  # P no centro
    for shell in range(1, n_shells + 1):
        scale = shell
        for dx in range(-shell, shell + 1):
            for dy in range(-shell, shell + 1):
                for dz in range(-shell, shell + 1):
                    if abs(dx) + abs(dy) + abs(dz) == 0:
                        continue
                    if max(abs(dx), abs(dy), abs(dz)) == shell:
                        # sítios tipo diamante
                        p = np.array([dx, dy, dz], dtype=float) * (a / 2)
                        pos.append(p)
                        # segundo sublattice
                        for v in nn:
                            pos.append(p + v * scale * 0.5)
    # único
    arr = np.array(pos)
    # dedup
    rounded = np.round(arr, 6)
    _, idx = np.unique(rounded, axis=0, return_index=True)
    arr = arr[np.sort(idx)]
    # garantir origem primeiro
    d0 = np.linalg.norm(arr, axis=1)
    order = np.argsort(d0)
    arr = arr[order]
    return arr, 0  # índice do P


def V_P_screened(
    R: np.ndarray,
    R_P: np.ndarray = None,
    eps_r: float = 11.7,
    r_core: float = 0.25,
    U_cc: float = -4.0,
    is_P: bool = False,
) -> float:
    """
    V_P(R) = −1/(ε_r |R−R_P|) (1 − e^{−|R−R_P|/r_core}) + U_cc δ_{i,P}
    """
    if R_P is None:
        R_P = np.zeros(3)
    r = np.linalg.norm(R - R_P)
    if r < 1e-12:
        v = U_cc if is_P else 0.0
        return float(v)
    v = -1.0 / (eps_r * r) * (1.0 - np.exp(-r / r_core))
    if is_P:
        v += U_cc
    return float(v)


def montar_hamiltoniano(
    pos: np.ndarray,
    idx_P: int,
    p: Dict,
) -> np.ndarray:
    """
    H de dimensão (N_at * 10) × (N_at * 10).
    Onsite + hopping NN (distância mínima).
    """
    n_at = len(pos)
    dim = n_at * N_ORB
    H = np.zeros((dim, dim), dtype=float)
    E_on = p["E_onsite"]
    hop = p["hop"]

    # distâncias
    dmat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    d_nn = np.min(dmat[dmat > 1e-8]) * 1.15  # cutoff NN

    for i in range(n_at):
        base_i = i * N_ORB
        Vp = V_P_screened(pos[i], pos[idx_P], p["eps_r"], p["r_core"], p["U_cc"], is_P=(i == idx_P))
        for a, name in enumerate(ORBITAIS):
            H[base_i + a, base_i + a] = E_on[name] + Vp

        for j in range(i + 1, n_at):
            if dmat[i, j] > d_nn:
                continue
            base_j = j * N_ORB
            # hoppings simplificados (s–s, s–p, p–p)
            # s-s
            H[base_i + 0, base_j + 0] = hop["ss"]
            H[base_j + 0, base_i + 0] = hop["ss"]
            # s-p
            for ap in range(1, 4):
                H[base_i + 0, base_j + ap] = hop["sp"]
                H[base_j + ap, base_i + 0] = hop["sp"]
                H[base_i + ap, base_j + 0] = hop["sp"]
                H[base_j + 0, base_i + ap] = hop["sp"]
            # p-p
            for ap in range(1, 4):
                for bp in range(1, 4):
                    val = hop["pp_sigma"] if ap == bp else hop["pp_pi"]
                    H[base_i + ap, base_j + bp] = val
                    H[base_j + bp, base_i + ap] = val
            # s*-s, s*-p
            H[base_i + 9, base_j + 0] = hop["s*s"]
            H[base_j + 0, base_i + 9] = hop["s*s"]
            for ap in range(1, 4):
                H[base_i + 9, base_j + ap] = hop["s*p"]
                H[base_j + ap, base_i + 9] = hop["s*p"]

    return H


def diagonalizar_tb(H: np.ndarray, n_estados: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Autovalores/autovetores (menores energias = estados doadores)."""
    evals, evecs = np.linalg.eigh(H)
    return evals[:n_estados], evecs[:, :n_estados]
