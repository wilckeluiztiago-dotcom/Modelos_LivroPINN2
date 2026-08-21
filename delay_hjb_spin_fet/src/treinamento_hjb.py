"""Treinamento PINN do Delay-HJB."""
import numpy as np
from typing import Dict, List, Optional
from .rede_pinn_hjb import RedePINN3D
from .hjbd_retardado import residuo_delay_hjb


def perda_hjb(
    rede: RedePINN3D,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    peso_pde: float = 1.0,
    peso_term: float = 10.0,
) -> tuple:
    V, Vt, Vx, Vxx = rede.derivadas(X_col)
    x, y = X_col[:, 0], X_col[:, 1]
    res = residuo_delay_hjb(V, Vt, Vx, Vxx, x, y)
    perda_pde = float(np.mean(res ** 2))
    pred_t = rede.prever(X_term)
    perda_term = float(np.mean((pred_t - V_term) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term


def treinar_delay_hjb(
    rede: RedePINN3D,
    X_col: np.ndarray,
    X_term: np.ndarray,
    V_term: np.ndarray,
    n_epocas: int = 400,
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
        p0, _, _ = perda_hjb(rede, X_col, X_term, V_term)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(40, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_hjb(rede, X_col, X_term, V_term)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, term = perda_hjb(rede, X_col, X_term, V_term)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | term={term:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
