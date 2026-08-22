"""Treinamento PINN TTM."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_ttm import RedePINN_TTM
from .residuo_ttm import perda_ttm
from .fisica_ttm import parametros_ttm_default


def treinar_ttm(
    rede: RedePINN_TTM,
    X_col: np.ndarray,
    X0: np.ndarray,
    Te0: np.ndarray,
    TL0: np.ndarray,
    p: Optional[Dict] = None,
    n_epocas: int = 300,
    taxa: float = 7e-4,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    if p is None:
        p = parametros_ttm_default()
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        p0, _, _ = perda_ttm(rede, X_col, X0, Te0, TL0, p)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(36, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_ttm(rede, X_col, X0, Te0, TL0, p)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, ic = perda_ttm(rede, X_col, X0, Te0, TL0, p)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | ic={ic:.4e}")
        if epoca % 120 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
