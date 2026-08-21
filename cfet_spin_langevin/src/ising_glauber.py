"""
Dinâmica de Glauber–Ising para ocupação discreta de sub-bandas.
Apêndice J.3 — magnetização / ocupação quântica discreta.
"""

import numpy as np
from typing import Optional, Tuple


class RedeIsingGlauber:
    """
    Rede de Ising 1D (ou cadeia) com dinâmica de Glauber.

    Spins σ_i ∈ {−1, +1} representam ocupação de sub-banda
    (ex.: +1 ocupada, −1 vazia) em cada sítio ao longo do canal
    ou entre canais empilhados do CFET.
    """

    def __init__(
        self,
        n_spins: int = 32,
        J_troca: float = 1.0,      # acoplamento de troca
        h_campo: float = 0.0,      # campo externo / polarização de gate
        beta: float = 1.0,         # 1/(kT) inverso
        semente: Optional[int] = 42,
    ):
        self.n_spins = n_spins
        self.J_troca = J_troca
        self.h_campo = h_campo
        self.beta = beta
        self.rng = np.random.default_rng(semente)
        self.spins = self.rng.choice([-1, 1], size=n_spins)

    def energia_local(self, i: int, spins: Optional[np.ndarray] = None) -> float:
        """Campo local em i: h_i = J (σ_{i-1}+σ_{i+1}) + h."""
        if spins is None:
            spins = self.spins
        n = self.n_spins
        viz = spins[(i - 1) % n] + spins[(i + 1) % n]
        return self.J_troca * viz + self.h_campo

    def passo_glauber(self) -> None:
        """
        Um passo de Glauber (Metropolis single-spin):
            P(flip) = 1 / (1 + exp(2 β σ_i h_i))
        """
        i = self.rng.integers(0, self.n_spins)
        h_i = self.energia_local(i)
        sigma = self.spins[i]
        p_flip = 1.0 / (1.0 + np.exp(2.0 * self.beta * sigma * h_i))
        if self.rng.random() < p_flip:
            self.spins[i] = -sigma

    def evoluir(self, n_passos: int) -> np.ndarray:
        """Evolui n_passos e retorna magnetização média temporal."""
        mags = np.zeros(n_passos)
        for t in range(n_passos):
            self.passo_glauber()
            mags[t] = np.mean(self.spins)
        return mags

    def magnetizacao(self) -> float:
        return float(np.mean(self.spins))

    def definir_campo(self, h: float) -> None:
        self.h_campo = h
