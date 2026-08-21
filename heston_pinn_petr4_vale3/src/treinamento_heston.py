"""Treinamento PINN Heston."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_heston import RedePINN_Heston
from .residuo_heston import perda_heston


def treinar_heston(
    rede: RedePINN_Heston,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    n_epocas: int = 300,
    taxa: float = 6e-4,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
    **kwargs,
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
        p0, _, _ = perda_heston(rede, X_col, X_term, V_term, **kwargs)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(36, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_heston(rede, X_col, X_term, V_term, **kwargs)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, term = perda_heston(rede, X_col, X_term, V_term, **kwargs)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | term={term:.4e}")
        if epoca % 120 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
