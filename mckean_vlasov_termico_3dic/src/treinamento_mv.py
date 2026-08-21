"""Treinamento PINN McKean–Vlasov."""
import numpy as np
from typing import Dict, List, Optional
from .rede_pinn_mv import RedePINN_MV
from .residuo_mv import perda_mv


def treinar_mv(
    rede: RedePINN_MV,
    X_col: np.ndarray,
    X0: np.ndarray,
    p0: np.ndarray,
    x_grade: np.ndarray,
    a: float = 1.2,
    sigma: float = 0.12,
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
        p0_loss, _, _ = perda_mv(rede, X_col, X0, p0, x_grade, a, sigma)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            pj, _, _ = perda_mv(rede, X_col, X0, p0, x_grade, a, sigma)
            grad[j] = (pj - p0_loss) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda, pde, ic = perda_mv(rede, X_col, X0, p0, x_grade, a, sigma)
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
