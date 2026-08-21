"""
Difusão de Langevin em potenciais não-lineares.
Apêndice J.3 — evolução contínua do potencial de canal.
"""

import numpy as np
from typing import Callable, Optional, Tuple


def potencial_nao_linear(
    phi: np.ndarray,
    a: float = 1.0,
    b: float = 0.3,
    c: float = 0.1,
) -> np.ndarray:
    """
    Potencial não-linear tipo double-well / Duffing:
        U(φ) = (a/2) φ² + (b/4) φ⁴ + c φ
    """
    return 0.5 * a * phi ** 2 + 0.25 * b * phi ** 4 + c * phi


def forca_potencial(
    phi: np.ndarray,
    a: float = 1.0,
    b: float = 0.3,
    c: float = 0.1,
) -> np.ndarray:
    """F = −∇U = −(a φ + b φ³ + c)."""
    return -(a * phi + b * phi ** 3 + c)


def passo_langevin(
    phi: np.ndarray,
    dt: float,
    gamma: float = 1.0,
    sigma: float = 0.15,
    a: float = 1.0,
    b: float = 0.3,
    c: float = 0.1,
    acoplamento_spin: float = 0.0,
    magnetizacao: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Euler–Maruyama para
        dφ = [ −γ ∇U(φ) + κ m ] dt + σ dW

    onde m é a magnetização Ising (ocupação de sub-banda)
    e κ o acoplamento espin–potencial (eletrostático / troca).
    """
    if rng is None:
        rng = np.random.default_rng()
    forca = forca_potencial(phi, a, b, c) * gamma + acoplamento_spin * magnetizacao
    dW = rng.normal(0.0, np.sqrt(dt), size=np.shape(phi))
    return phi + forca * dt + sigma * dW


class ProcessoLangevin:
    """Processo de Langevin escalar ou vetorial (canais n/p)."""

    def __init__(
        self,
        phi0: float = 0.0,
        gamma: float = 1.0,
        sigma: float = 0.15,
        a: float = 1.0,
        b: float = 0.3,
        c: float = 0.1,
        semente: Optional[int] = 0,
    ):
        self.phi = float(phi0)
        self.gamma = gamma
        self.sigma = sigma
        self.a = a
        self.b = b
        self.c = c
        self.rng = np.random.default_rng(semente)

    def passo(self, dt: float, acoplamento_spin: float = 0.0, m: float = 0.0) -> float:
        self.phi = float(
            passo_langevin(
                np.array([self.phi]),
                dt,
                self.gamma,
                self.sigma,
                self.a,
                self.b,
                self.c,
                acoplamento_spin,
                m,
                self.rng,
            )[0]
        )
        return self.phi
