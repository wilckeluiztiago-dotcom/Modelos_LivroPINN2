"""Treinamento PINN–PIDE."""
import numpy as np
from typing import Dict, List, Optional
from .rede_pinn import RedePINN
from .barreira_tunelamento import CanalSub12nm
from .residuo_pide import perda_pide


def treinar_pide(
    rede: RedePINN,
    x_col: np.ndarray,
    x_bc: np.ndarray,
    n_bc: np.ndarray,
    canal: CanalSub12nm,
    n_epocas: int = 400,
    taxa: float = 5e-4,
    peso_pde: float = 1.0,
    peso_bc: float = 15.0,
    n_mc: int = 16,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico: List[float] = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    beta1 = 0.9
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        seed_mc = int(g.integers(0, 1_000_000))
        perda0, _, _ = perda_pide(
            rede, x_col, x_bc, n_bc, canal, peso_pde, peso_bc, n_mc, seed_mc
        )
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(36, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_pide(
                rede, x_col, x_bc, n_bc, canal, peso_pde, peso_bc, n_mc, seed_mc
            )
            grad[j] = (pj - perda0) / eps_g

        m = beta1 * m + (1 - beta1) * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)

        perda, pde, bc = perda_pide(
            rede, x_col, x_bc, n_bc, canal, peso_pde, peso_bc, n_mc, seed_mc
        )
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()

        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e} | pde={pde:.4e} | bc={bc:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor, "n_parametros": n_params}
