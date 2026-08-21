"""
Dinâmica híbrida LSV (Dupire–Heston adaptada a portadores).

    dE_t = ... dt + L(E_t, t) √ν_t  dW   (campo / velocidade efetiva)
    dν_t = κ(θ − ν) dt + ξ √ν dZ
"""

import numpy as np
from typing import Optional, Dict, Tuple
from .cir_variancia import passo_cir
from .fator_local import fator_local_L, mobilidade_efetiva_dupire


def passo_lsv_portador(
    E: float,
    nu: float,
    dt: float,
    L_fn,
    kappa: float = 2.0,
    theta: float = 1.0,
    xi: float = 0.5,
    drift_E: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """
    Euler–Maruyama acoplado:
        dE = drift_E dt + L(E) √ν dW
        dν = CIR
    """
    if rng is None:
        rng = np.random.default_rng()
    L = float(L_fn(np.array([E]))[0])
    dW = rng.normal(0.0, np.sqrt(dt))
    E_new = E + drift_E * dt + L * np.sqrt(max(nu, 0.0)) * dW
    nu_new = passo_cir(nu, dt, kappa, theta, xi, rng)
    return float(E_new), float(nu_new)


def simular_lsv(
    n_passos: int = 2000,
    dt: float = 0.005,
    E0: float = 0.5,
    nu0: float = 1.0,
    kappa: float = 2.0,
    theta: float = 1.0,
    xi: float = 0.4,
    mu0: float = 1.0,
    E_sat: float = 1.5,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    """Simula trajetória LSV de campo efetivo E_t e variância ν_t."""
    g = np.random.default_rng(semente)

    def L_fn(Earr):
        return fator_local_L(Earr, mu0=mu0, E_sat=E_sat, E_nu=np.ones_like(Earr) * theta)

    E = np.zeros(n_passos + 1)
    nu = np.zeros(n_passos + 1)
    E[0], nu[0] = E0, nu0
    for k in range(n_passos):
        E[k + 1], nu[k + 1] = passo_lsv_portador(
            E[k], nu[k], dt, L_fn, kappa, theta, xi, drift_E=0.05, rng=g
        )
    t = np.arange(n_passos + 1) * dt
    return {"t": t, "E": E, "nu": nu}
