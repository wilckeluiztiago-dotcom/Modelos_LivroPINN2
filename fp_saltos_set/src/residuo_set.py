"""
Resíduo Kolmogorov Forward híbrido:

∂p/∂t − (σ²/2) ∂²p/∂s²
  = λ^a(s) p(q−1,s,t) + λ^b(s) p(q+1,s,t) − (λ^a+λ^b) p(q,s,t)

(Nota: convenção de índices: λ^a adiciona elétron q→q+1, então
 a massa em q recebe de q−1 via λ^a e de q+1 via λ^b.)
"""

import numpy as np
from typing import Tuple
from .rede_pinn_set import RedePINN_SET
from .set_carga import taxas_tunelamento


def residuo_kolmogorov(
    rede: RedePINN_SET,
    q: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    q_max: float = 5.0,
    sigma: float = 0.15,
    Gamma0: float = 1.0,
    E_c: float = 0.5,
    V_bias: float = 0.3,
) -> np.ndarray:
    X = np.column_stack([q / q_max, s, t])
    p, pt, ps, pss = rede.derivadas_s_t(X)

    # p nos vizinhos q±1
    Xp = np.column_stack([(q + 1) / q_max, s, t])
    Xm = np.column_stack([(q - 1) / q_max, s, t])
    p_qp1 = rede.prever(Xp)
    p_qm1 = rede.prever(Xm)

    la, lb = taxas_tunelamento(s, 0, Gamma0, E_c=E_c, V_bias=V_bias)  # base; depende de s
    # taxas dependem do estado de origem; aproximamos com q local
    la_m, _ = taxas_tunelamento(s, 0, Gamma0, E_c=E_c, V_bias=V_bias)
    # forma simplificada: λ^a(s), λ^b(s) funções de s
    jump = la * p_qm1 + lb * p_qp1 - (la + lb) * p
    return pt - 0.5 * sigma ** 2 * pss - jump


def perda_set(
    rede: RedePINN_SET,
    q: np.ndarray,
    s: np.ndarray,
    t: np.ndarray,
    q0: np.ndarray,
    s0: np.ndarray,
    p0: np.ndarray,
    sigma: float = 0.15,
    peso_pde: float = 1.0,
    peso_ic: float = 10.0,
) -> Tuple[float, float, float]:
    res = residuo_kolmogorov(rede, q, s, t, sigma=sigma)
    perda_pde = float(np.mean(res ** 2))
    X0 = np.column_stack([q0 / 5.0, s0, np.zeros_like(s0)])
    pred0 = rede.prever(X0)
    perda_ic = float(np.mean((pred0 - p0) ** 2))
    return peso_pde * perda_pde + peso_ic * perda_ic, perda_pde, perda_ic
