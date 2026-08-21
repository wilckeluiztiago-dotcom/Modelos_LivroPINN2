"""Treinamento PINN PIDE energia/PLD."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_pide import RedePINN_PIDE
from .residuo_pide import perda_pide


def treinar_pide(
    rede: RedePINN_PIDE,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    n_epocas: int = 300,
    taxa: float = 7e-4,
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
        seed_mc = int(g.integers(0, 1e6))
        rng_mc = np.random.default_rng(seed_mc)
        p0, _, _ = perda_pide(rede, X_col, X_term, V_term, rng=rng_mc, n_mc=8, **kwargs)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_pide(rede, X_col, X_term, V_term, rng=rng_mc, n_mc=8, **kwargs)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, term = perda_pide(rede, X_col, X_term, V_term, rng=rng_mc, n_mc=8, **kwargs)
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
