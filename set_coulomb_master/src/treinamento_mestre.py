"""Treinamento PINN equação mestre SET."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_mestre import RedePINN_Mestre
from .residuo_mestre import perda_mestre


def treinar_mestre(
    rede: RedePINN_Mestre,
    N_vals: np.ndarray,
    t_col: np.ndarray,
    P0: np.ndarray,
    V_g: float,
    V_sd: float,
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
        p0, _, _ = perda_mestre(rede, N_vals, t_col, P0, V_g, V_sd, **kwargs)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_mestre(rede, N_vals, t_col, P0, V_g, V_sd, **kwargs)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, ic = perda_mestre(rede, N_vals, t_col, P0, V_g, V_sd, **kwargs)
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
