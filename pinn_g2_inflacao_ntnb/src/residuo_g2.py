"""
Resíduo da EDP de precificação a dois fatores (forma reduzida G2++):

  ∂P/∂τ = (r+i) P − κ_r(θ_r−r) ∂P/∂r − κ_i(θ_i−i) ∂P/∂i
          + (1/2)σ_r² ∂²P/∂r² + (1/2)σ_i² ∂²P/∂i²
          + ρ σ_r σ_i ∂²P/∂r∂i

(aqui usamos τ = maturidade residual; sinal conforme convenção).
"""

import numpy as np
from typing import Tuple
from .rede_pinn_g2 import RedePINN_G2


def residuo_edp_nominal(
    rede: RedePINN_G2,
    X: np.ndarray,
    kappa_r: float = 0.3,
    theta_r: float = 0.04,
    sigma_r: float = 0.01,
    kappa_i: float = 0.5,
    theta_i: float = 0.045,
    sigma_i: float = 0.008,
    rho: float = 0.3,
) -> np.ndarray:
    P, Pr, Pi, Ptau, Prr, Pii = rede.derivadas(X)
    r, i = X[:, 0], X[:, 1]
    # resíduo: ∂P/∂τ − (r+i)P + drifts·grads − (1/2) vols·hess
    # forma: L[P] = 0 com L = ∂τ − (r+i) + μ_r ∂r + μ_i ∂i + (1/2)σ²∂rr + ...
    # usamos: P_τ + (r+i)P − μ_r Pr − μ_i Pi − 0.5 σ_r² Prr − 0.5 σ_i² Pii ≈ 0
    mu_r = kappa_r * (theta_r - r)
    mu_i = kappa_i * (theta_i - i)
    return (
        Ptau
        + (r + i) * P
        - mu_r * Pr
        - mu_i * Pi
        - 0.5 * sigma_r ** 2 * Prr
        - 0.5 * sigma_i ** 2 * Pii
    )


def perda_g2(
    rede: RedePINN_G2,
    X_col: np.ndarray,
    X_term: np.ndarray,
    peso_pde: float = 1.0,
    peso_term: float = 15.0,
    **kwargs,
) -> Tuple[float, float, float]:
    res = residuo_edp_nominal(rede, X_col, **kwargs)
    perda_pde = float(np.mean(res ** 2))
    # P(r,i,0) = 1
    pred_term = rede.prever(X_term)
    perda_term = float(np.mean((pred_term - 1.0) ** 2))
    return peso_pde * perda_pde + peso_term * perda_term, perda_pde, perda_term
