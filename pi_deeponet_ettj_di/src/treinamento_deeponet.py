"""Treinamento PI-DeepONet para ETTJ DI."""
import numpy as np
from typing import Dict, Optional
from .rede_deeponet import PIDeepONet
from .residuo_hjm import perda_deeponet


def treinar_deeponet(
    rede: PIDeepONet,
    curva: np.ndarray,
    tT_dados: np.ndarray,
    P_dados: np.ndarray,
    tT_col: np.ndarray,
    r_curto: float = 0.12,
    n_epocas: int = 400,
    taxa: float = 8e-4,
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
        p0, _, _ = perda_deeponet(rede, curva, tT_dados, P_dados, tT_col, r_curto)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(40, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_deeponet(rede, curva, tT_dados, P_dados, tT_col, r_curto)
            grad[j] = (pj - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, dados, pde = perda_deeponet(rede, curva, tT_dados, P_dados, tT_col, r_curto)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | dados={dados:.4e} | pde={pde:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
