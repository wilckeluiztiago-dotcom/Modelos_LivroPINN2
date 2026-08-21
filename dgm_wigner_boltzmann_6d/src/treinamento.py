"""
Treinamento DGM para Wigner–Boltzmann.
"""

import numpy as np
from typing import Dict, List, Optional
from .celula_dgm import RedeDGM
from .wigner_boltzmann import NanofolhaWigner
from .residuo_wigner import perda_composta_wigner


def treinar_dgm(
    rede: RedeDGM,
    X_col: np.ndarray,
    X0: np.ndarray,
    f0: np.ndarray,
    nano: NanofolhaWigner,
    n_epocas: int = 400,
    taxa: float = 5e-4,
    peso_pde: float = 1.0,
    peso_ic: float = 12.0,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    gerador = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico: List[float] = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    beta1 = 0.9
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        perda0, _, _ = perda_composta_wigner(
            rede, X_col, X0, f0, nano, peso_pde, peso_ic
        )
        grad = np.zeros_like(theta)
        idx = gerador.choice(n_params, size=min(40, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_composta_wigner(
                rede, X_col, X0, f0, nano, peso_pde, peso_ic
            )
            grad[j] = (pj - perda0) / eps_g

        m = beta1 * m + (1 - beta1) * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)

        perda, pde, ic = perda_composta_wigner(
            rede, X_col, X0, f0, nano, peso_pde, peso_ic
        )
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()

        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | ic={ic:.4e}")

        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor, "n_parametros": n_params}
