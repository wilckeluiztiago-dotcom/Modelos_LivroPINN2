"""Treinamento PINN Fokker–Planck."""
import numpy as np
from typing import Dict, List, Optional
from .rede_pinn_fp import RedePINN_FP
from .residuo_fp import perda_fp


def treinar_fp(
    rede: RedePINN_FP,
    X_col: np.ndarray,
    X0: np.ndarray,
    p0: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
    n_epocas: int = 350,
    taxa: float = 7e-4,
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
        p0_loss, _, _ = perda_fp(rede, X_col, X0, p0, a, b, sigma)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(36, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_fp(rede, X_col, X0, p0, a, b, sigma)
            grad[j] = (pj - p0_loss) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, ic = perda_fp(rede, X_col, X0, p0, a, b, sigma)
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
