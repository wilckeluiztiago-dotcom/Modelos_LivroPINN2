"""Treinamento PINN Kolmogorov SET."""
import numpy as np
from typing import Dict, Optional
from .rede_pinn_set import RedePINN_SET
from .residuo_set import perda_set


def treinar_set(
    rede: RedePINN_SET,
    q, s, t, q0, s0, p0,
    sigma: float = 0.15,
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
        p0_loss, _, _ = perda_set(rede, q, s, t, q0, s0, p0, sigma)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_set(rede, q, s, t, q0, s0, p0, sigma)
            grad[j] = (pj - p0_loss) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, ic = perda_set(rede, q, s, t, q0, s0, p0, sigma)
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
