"""
Resíduo do operador de não-arbitragem (forma reduzida HJM / PDE do título):

    ∂P/∂t + r_t P − f(t,T) P + (1/2) σ_P²(t,T) P ≈ 0

Em curva estática (σ→0, f≈r curto para T↓t):
    ∂P/∂t − f(0,T)·(algo) ... forma simplificada usada no treino:

    ∂P/∂t + r(t) P ≈ 0   quando T fixo e sem vol,
    com condição P(T,T)=1.
"""

import numpy as np
from typing import Tuple
from .rede_deeponet import PIDeepONet


def residuo_pde_titulo(
    rede: PIDeepONet,
    curva: np.ndarray,
    tT: np.ndarray,
    r_curto: float = 0.12,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Resíduo aproximado: ∂P/∂t + r_curto · P  (sem vol, curva estática).
    """
    tT = np.asarray(tT, dtype=float)
    P = rede.prever(curva, tT)
    tTp = tT.copy()
    tTp[:, 0] += eps
    Pp = rede.prever(curva, tTp)
    Pt = (Pp - P) / eps
    return Pt + r_curto * P


def perda_deeponet(
    rede: PIDeepONet,
    curva: np.ndarray,
    tT_dados: np.ndarray,
    P_dados: np.ndarray,
    tT_col: np.ndarray,
    r_curto: float = 0.12,
    peso_dados: float = 1.0,
    peso_pde: float = 0.3,
    peso_terminal: float = 5.0,
) -> Tuple[float, float, float]:
    pred = rede.prever(curva, tT_dados)
    perda_d = float(np.mean((pred - P_dados) ** 2))
    res = residuo_pde_titulo(rede, curva, tT_col, r_curto)
    perda_pde = float(np.mean(res ** 2))
    # P(T,T) = 1
    n_term = min(30, len(tT_dados))
    Tvals = tT_dados[:n_term, 1]
    tT_term = np.column_stack([Tvals, Tvals])
    pred_term = rede.prever(curva, tT_term)
    perda_term = float(np.mean((pred_term - 1.0) ** 2))
    total = peso_dados * perda_d + peso_pde * perda_pde + peso_terminal * perda_term
    return total, perda_d, perda_pde
