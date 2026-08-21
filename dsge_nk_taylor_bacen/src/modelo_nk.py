"""
DSGE Neo-Keynesiano 3 equações + Regra de Taylor (estilo Copom/Bacen).

  IS:       ŷ_t = E_t[ŷ_{t+1}] − (1/σ)(î_t − E_t[π̂_{t+1}] − r̂^n_t)
  Phillips: π̂_t = β E_t[π̂_{t+1}] + κ ŷ_t
  Taylor:   î_t = φ_π π̂_t + φ_y ŷ_t + ε^i_t
"""

import numpy as np
from typing import Optional, Dict, Tuple


class ParametrosNK:
    def __init__(
        self,
        sigma: float = 1.0,      # elasticidade intertemporal
        beta: float = 0.99,      # desconto
        kappa: float = 0.1,      # slope Phillips
        phi_pi: float = 1.5,     # Taylor: peso inflação (Taylor principle)
        phi_y: float = 0.125,    # Taylor: peso produto
        rho_rn: float = 0.8,     # persistência taxa natural
        rho_tt: float = 0.7,     # termos de troca
        rho_fisc: float = 0.6,   # prêmio fiscal
        sig_rn: float = 0.01,
        sig_tt: float = 0.008,
        sig_fisc: float = 0.005,
        sig_i: float = 0.002,    # choque monetário
    ):
        self.sigma = sigma
        self.beta = beta
        self.kappa = kappa
        self.phi_pi = phi_pi
        self.phi_y = phi_y
        self.rho_rn = rho_rn
        self.rho_tt = rho_tt
        self.rho_fisc = rho_fisc
        self.sig_rn = sig_rn
        self.sig_tt = sig_tt
        self.sig_fisc = sig_fisc
        self.sig_i = sig_i


def solucao_estatica_nk(
    rn: float,
    tt: float,
    fisc: float,
    eps_i: float,
    p: ParametrosNK,
) -> Tuple[float, float, float]:
    """
    Solução sob expectativas racionais com choques AR(1) conhecidos
    (método de coeficientes indeterminados simplificado, horizonte 1).

    Aproxima E[y_{t+1}] ≈ ρ_eff · y_t, etc. para fechar o sistema linear.
    """
    # sistema linear 3x3 em (y, pi, i)
    # y = ρ_y y − (1/σ)(i − ρ_π π − rn − tt)   [tt entra como demanda]
    # π = β ρ_π π + κ y
    # i = φ_π π + φ_y y + eps_i + fisc
    #
    # Usamos expectativa miópica de persistência média ρ̄
    rho_bar = 0.5 * (p.rho_rn + p.rho_tt)
    s = p.sigma
    # Matriz A · [y, π, i]^T = b
    # Eq1: (1−ρ_bar) y + (1/s) i − (ρ_bar/s) π = (1/s)(rn + tt)
    # Eq2: −κ y + (1 − β ρ_bar) π = 0
    # Eq3: −φ_y y − φ_π π + i = eps_i + fisc
    A = np.array([
        [1 - rho_bar, -rho_bar / s, 1.0 / s],
        [-p.kappa, 1 - p.beta * rho_bar, 0.0],
        [-p.phi_y, -p.phi_pi, 1.0],
    ])
    b = np.array([
        (rn + tt) / s,
        0.0,
        eps_i + fisc,
    ])
    try:
        sol = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol = np.zeros(3)
    return float(sol[0]), float(sol[1]), float(sol[2])


def simular_dsge(
    n_periodos: int = 80,
    p: Optional[ParametrosNK] = None,
    semente: Optional[int] = 42,
) -> Dict[str, np.ndarray]:
    if p is None:
        p = ParametrosNK()
    g = np.random.default_rng(semente)
    y = np.zeros(n_periodos)
    pi = np.zeros(n_periodos)
    i = np.zeros(n_periodos)
    rn = np.zeros(n_periodos)
    tt = np.zeros(n_periodos)
    fisc = np.zeros(n_periodos)

    for t in range(n_periodos):
        if t == 0:
            rn[t] = p.sig_rn * g.normal()
            tt[t] = p.sig_tt * g.normal()
            fisc[t] = p.sig_fisc * g.normal()
        else:
            rn[t] = p.rho_rn * rn[t - 1] + p.sig_rn * g.normal()
            tt[t] = p.rho_tt * tt[t - 1] + p.sig_tt * g.normal()
            fisc[t] = p.rho_fisc * fisc[t - 1] + p.sig_fisc * g.normal()
        eps_i = p.sig_i * g.normal()
        y[t], pi[t], i[t] = solucao_estatica_nk(rn[t], tt[t], fisc[t], eps_i, p)

    return {
        "t": np.arange(n_periodos),
        "y": y,
        "pi": pi,
        "i": i,
        "rn": rn,
        "tt": tt,
        "fisc": fisc,
    }


def impulso_resposta(
    choque: str = "rn",
    tamanho: float = 0.01,
    n_periodos: int = 40,
    p: Optional[ParametrosNK] = None,
) -> Dict[str, np.ndarray]:
    """IRF determinística a um choque unitário em t=0."""
    if p is None:
        p = ParametrosNK()
    y = np.zeros(n_periodos)
    pi = np.zeros(n_periodos)
    i = np.zeros(n_periodos)
    rn = np.zeros(n_periodos)
    tt = np.zeros(n_periodos)
    fisc = np.zeros(n_periodos)

    if choque == "rn":
        rn[0] = tamanho
    elif choque == "tt":
        tt[0] = tamanho
    elif choque == "fisc":
        fisc[0] = tamanho
    elif choque == "i":
        pass  # tratado via eps

    for t in range(n_periodos):
        if t > 0:
            rn[t] = p.rho_rn * rn[t - 1]
            tt[t] = p.rho_tt * tt[t - 1]
            fisc[t] = p.rho_fisc * fisc[t - 1]
        eps = tamanho if (choque == "i" and t == 0) else 0.0
        y[t], pi[t], i[t] = solucao_estatica_nk(rn[t], tt[t], fisc[t], eps, p)

    return {"t": np.arange(n_periodos), "y": y, "pi": pi, "i": i}
