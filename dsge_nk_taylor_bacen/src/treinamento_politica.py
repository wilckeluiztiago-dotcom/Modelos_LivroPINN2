"""
Treino supervisionado + resíduo de equilíbrio das 3 equações NK.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .rede_politica import RedePoliticaNK
from .modelo_nk import ParametrosNK, solucao_estatica_nk


def residuo_equilibrio(
    y: np.ndarray,
    pi: np.ndarray,
    i: np.ndarray,
    rn: np.ndarray,
    tt: np.ndarray,
    p: ParametrosNK,
    rho_bar: float = 0.75,
) -> np.ndarray:
    """
    Resíduos das 3 equações sob expectativa E[x_{t+1}] ≈ ρ̄ x_t.
    """
    s = p.sigma
    # IS
    res_is = y - rho_bar * y + (1.0 / s) * (i - rho_bar * pi - rn - tt)
    # Phillips
    res_ph = pi - p.beta * rho_bar * pi - p.kappa * y
    # Taylor
    res_ty = i - p.phi_pi * pi - p.phi_y * y
    return np.column_stack([res_is, res_ph, res_ty])


def perda_politica(
    rede: RedePoliticaNK,
    estados: np.ndarray,
    alvos: np.ndarray,
    p: ParametrosNK,
    peso_dados: float = 1.0,
    peso_eq: float = 0.5,
) -> Tuple[float, float, float]:
    pred = rede.prever(estados)
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)
    perda_d = float(np.mean((pred - alvos) ** 2))
    rn, tt, fisc = estados[:, 0], estados[:, 1], estados[:, 2]
    res = residuo_equilibrio(pred[:, 0], pred[:, 1], pred[:, 2], rn, tt, p)
    perda_eq = float(np.mean(res ** 2))
    return peso_dados * perda_d + peso_eq * perda_eq, perda_d, perda_eq


def treinar_politica(
    rede: RedePoliticaNK,
    estados: np.ndarray,
    alvos: np.ndarray,
    p: ParametrosNK,
    n_epocas: int = 400,
    taxa: float = 8e-4,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        p0, _, _ = perda_politica(rede, estados, alvos, p)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(40, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_politica(rede, estados, alvos, p)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, d, eq = perda_politica(rede, estados, alvos, p)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | dados={d:.4e} | eq={eq:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
