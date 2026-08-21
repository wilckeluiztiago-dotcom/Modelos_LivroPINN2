"""
Modelo de dois fatores: taxa real r_t e inflação instantânea i_t (estilo G2++).
NTN-B (IPCA+) vs DI/LTN (pré).
"""

import numpy as np
from typing import Optional, Dict, Tuple


def passo_dois_fatores(
    r: float,
    i: float,
    dt: float,
    kappa_r: float = 0.3,
    theta_r: float = 0.04,
    sigma_r: float = 0.01,
    kappa_i: float = 0.5,
    theta_i: float = 0.045,
    sigma_i: float = 0.008,
    rho: float = 0.3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    dr = κ_r (θ_r − r) dt + σ_r dW^r
    di = κ_i (θ_i − i) dt + σ_i dW^i
    corr(dW^r, dW^i) = ρ
    """
    if rng is None:
        rng = np.random.default_rng()
    z1 = rng.normal()
    z2 = rho * z1 + np.sqrt(1 - rho ** 2) * rng.normal()
    r_new = r + kappa_r * (theta_r - r) * dt + sigma_r * np.sqrt(dt) * z1
    i_new = i + kappa_i * (theta_i - i) * dt + sigma_i * np.sqrt(dt) * z2
    return float(r_new), float(i_new)


def simular_fatores(
    n_passos: int = 1000,
    dt: float = 0.01,
    r0: float = 0.04,
    i0: float = 0.045,
    semente: Optional[int] = 42,
    **kwargs,
) -> Dict[str, np.ndarray]:
    g = np.random.default_rng(semente)
    r = np.zeros(n_passos + 1)
    i = np.zeros(n_passos + 1)
    r[0], i[0] = r0, i0
    for k in range(n_passos):
        r[k + 1], i[k + 1] = passo_dois_fatores(r[k], i[k], dt, rng=g, **kwargs)
    t = np.arange(n_passos + 1) * dt
    # taxa nominal ≈ real + inflação (Fisher linearizado)
    n = r + i
    return {"t": t, "r": r, "i": i, "n": n}


def preco_nominal_analitico_approx(
    r: float,
    i: float,
    tau: float,
    kappa_r: float = 0.3,
    theta_r: float = 0.04,
) -> float:
    """
    Aproximação: P_nom ≈ exp(−(r+i)·τ) sob taxas congeladas.
    """
    return float(np.exp(-(r + i) * tau))


def preco_real_analitico_approx(r: float, tau: float) -> float:
    """P_real ≈ exp(−r·τ)."""
    return float(np.exp(-r * tau))


def inflacao_implicita_breakeven(
    P_nom: float,
    P_real: float,
    tau: float,
) -> float:
    """
    Breakeven: (1 + i_be)^τ = P_real / P_nom
    → i_be = (P_real / P_nom)^{1/τ} − 1
    """
    if tau < 1e-8 or P_nom < 1e-12:
        return 0.0
    ratio = max(P_real / P_nom, 1e-12)
    return float(ratio ** (1.0 / tau) - 1.0)
