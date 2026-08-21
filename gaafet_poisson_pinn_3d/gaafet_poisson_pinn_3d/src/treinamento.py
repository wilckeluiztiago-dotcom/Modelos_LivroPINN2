"""
Treinamento híbrido conceitual (Adam-like + refinamento).
Cap. 3.6 do livro — Otimização Híbrida Adam e L-BFGS.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from .rede_pinn3d import RedePINN3D
from .residuo_poisson import perda_composta


def treinar_pinn3d(
    rede: RedePINN3D,
    X_col: np.ndarray,
    X_bc: np.ndarray,
    valores_bc: np.ndarray,
    epsilon_fn: Callable,
    rho_fn: Callable,
    n_epocas: int = 600,
    taxa: float = 5e-4,
    peso_pde: float = 1.0,
    peso_bc: float = 15.0,
    semente: Optional[int] = 0,
    verbose_cada: int = 100,
) -> Dict:
    """
    Gradiente estocástico por diferenças finitas no espaço de parâmetros
    (implementação didática do fluxo Cap. 2.7 / 3.6).
    """
    gerador = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico: List[float] = []
    melhor = np.inf
    melhor_theta = theta.copy()
    eps_g = 1e-5
    m = np.zeros_like(theta)  # momento estilo Adam simplificado
    beta1 = 0.9

    for epoca in range(1, n_epocas + 1):
        perda0, _, _ = perda_composta(
            rede, X_col, X_bc, valores_bc, epsilon_fn, rho_fn, peso_pde, peso_bc
        )
        grad = np.zeros_like(theta)
        # subamostra de parâmetros para eficiência
        idx = gerador.choice(n_params, size=min(48, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_composta(
                rede, X_col, X_bc, valores_bc, epsilon_fn, rho_fn, peso_pde, peso_bc
            )
            grad[j] = (pj - perda0) / eps_g

        m = beta1 * m + (1 - beta1) * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)

        perda, pde, bc = perda_composta(
            rede, X_col, X_bc, valores_bc, epsilon_fn, rho_fn, peso_pde, peso_bc
        )
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()

        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | bc={bc:.4e}")

        if epoca % 200 == 0:
            taxa *= 0.75

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor, "n_parametros": n_params}
