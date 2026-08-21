"""Treinamento PINN para condição de Gyöngy."""
import numpy as np
from typing import Dict, List, Optional
from .rede_pinn_gyongy import RedePINN1D
from .gyongy_calibracao import perda_gyongy_pinn


def treinar_gyongy(
    rede: RedePINN1D,
    grade_E: np.ndarray,
    E_nu: np.ndarray,
    n_epocas: int = 400,
    taxa: float = 8e-4,
    mu0: float = 1.0,
    E_sat: float = 1.5,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico: List[float] = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        perda0 = perda_gyongy_pinn(rede, grade_E, E_nu, mu0, E_sat)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            grad[j] = (perda_gyongy_pinn(rede, grade_E, E_nu, mu0, E_sat) - perda0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda = perda_gyongy_pinn(rede, grade_E, E_nu, mu0, E_sat)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda Gyöngy={perda:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
