"""Treinamento variacional da B-PINN."""
import numpy as np
from typing import Dict, List, Optional
from .rede_bpinn import RedeBayesiana
from .residuo_bpinn import elbo_bpinn


def treinar_bpinn(
    rede: RedeBayesiana,
    x_col: np.ndarray,
    rho: np.ndarray,
    x_bc: np.ndarray,
    v_bc: np.ndarray,
    n_epocas: int = 350,
    taxa: float = 5e-4,
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
        p0, _, _ = elbo_bpinn(rede, x_col, rho, x_bc, v_bc)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(36, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = elbo_bpinn(rede, x_col, rho, x_bc, v_bc)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        # clip log-sigma
        rede.carregar_parametros(theta)
        for i in range(rede.n_camadas):
            rede.log_sigma_pesos[i] = np.clip(rede.log_sigma_pesos[i], -6, 0)
            rede.log_sigma_vieses[i] = np.clip(rede.log_sigma_vieses[i], -6, 0)
        theta = rede.parametros_vetor().copy()

        perda, pde, kl = elbo_bpinn(rede, x_col, rho, x_bc, v_bc)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | −ELBO={perda:.4e} | pde={pde:.4e} | KL={kl:.3f}")
        if epoca % 120 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
