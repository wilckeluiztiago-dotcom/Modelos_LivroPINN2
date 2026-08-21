"""Treinamento PINN Feynman–Kac com saltos."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_fk import RedePINN_FK
from .residuo_fk import perda_fk


def treinar_fk(
    rede: RedePINN_FK,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    r: float = 0.05,
    sigma: float = 0.25,
    lam: float = 0.8,
    n_epocas: int = 300,
    taxa: float = 6e-4,
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
        seed_mc = int(g.integers(0, 1e6))
        p0, _, _ = perda_fk(rede, X_col, X_term, V_term, r, sigma, lam, n_mc=8, semente=seed_mc)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_fk(rede, X_col, X_term, V_term, r, sigma, lam, n_mc=8, semente=seed_mc)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, term = perda_fk(rede, X_col, X_term, V_term, r, sigma, lam, n_mc=8, semente=seed_mc)
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
