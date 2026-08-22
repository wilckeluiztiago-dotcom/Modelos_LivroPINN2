"""
Física do Single-Electron Transistor (SET) e Bloqueio de Coulomb.

Condição: E_c = e²/(2 C_Σ) ≫ k_B T
"""

import numpy as np
from typing import Tuple, Optional


# constantes efetivas (unidades normalizadas)
e = 1.0          # carga elementar
kB = 1.0         # Boltzmann (unidades onde kB=1)


def energia_carregamento(C_Sigma: float = 1.0) -> float:
    """E_c = e² / (2 C_Σ)"""
    return (e ** 2) / (2.0 * C_Sigma)


def energia_livre(
    N: int,
    V_g: float,
    V_sd: float,
    C_g: float = 0.3,
    C_Sigma: float = 1.0,
    n_g0: float = 0.0,
) -> float:
    """
    Energia eletrostática da ilha com N elétrons:

        U(N) = E_c (N − n_g)² − α N V_sd

    n_g = C_g V_g / e + n_g0  (número de carga induzido pelo gate)
    """
    E_c = energia_carregamento(C_Sigma)
    n_g = C_g * V_g / e + n_g0
    return E_c * (N - n_g) ** 2 - 0.1 * N * V_sd


def taxas_tunelamento(
    N: int,
    V_g: float,
    V_sd: float,
    Gamma0: float = 1.0,
    T: float = 0.05,
    C_g: float = 0.3,
    C_Sigma: float = 1.0,
) -> Tuple[float, float]:
    """
    Taxas de adição (N→N+1) e remoção (N→N−1) via regra de ouro de Fermi
    com fator de Fermi-Dirac térmico.

        ΔE_+ = U(N+1) − U(N)
        Γ_+ = Γ0 · ΔE_+ / (e^{ΔE_+/kT} − 1)   (forma simplificada)

    Em T→0: Γ_+ > 0 só se ΔE_+ < 0 (energia diminui ao adicionar).
    """
    U_N = energia_livre(N, V_g, V_sd, C_g, C_Sigma)
    U_Np = energia_livre(N + 1, V_g, V_sd, C_g, C_Sigma)
    U_Nm = energia_livre(N - 1, V_g, V_sd, C_g, C_Sigma)
    dE_add = U_Np - U_N      # energia para adicionar elétron
    dE_rem = U_N - U_Nm      # energia liberada ao remover (ou custo)

    def gamma(dE: float) -> float:
        # taxa ~ Γ0 / (1 + exp(dE / kT))  — permite tunelamento se dE ≲ kT
        return Gamma0 / (1.0 + np.exp(np.clip(dE / max(T, 1e-8), -40, 40)))

    Gamma_add = gamma(dE_add)   # N → N+1
    Gamma_rem = gamma(-dE_rem)  # N → N−1  (se remover baixa energia, -dE_rem < 0)
    # ajuste: remoção favorecida se U(N-1) < U(N) ⇒ dE_rem > 0 ⇒ -dE_rem < 0
    Gamma_rem = gamma(U_Nm - U_N)
    return float(Gamma_add), float(Gamma_rem)


def corrente_media(
    P: np.ndarray,
    N_vals: np.ndarray,
    V_g: float,
    V_sd: float,
    **kwargs,
) -> float:
    """I ∝ Σ_N [Γ_add(N) − Γ_rem(N)] P(N)"""
    I = 0.0
    for i, N in enumerate(N_vals):
        ga, gr = taxas_tunelamento(int(N), V_g, V_sd, **kwargs)
        I += (ga - gr) * P[i]
    return float(I)
