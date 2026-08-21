"""
Dinâmica de magnetização / polarização de spin com retardo τ_spin.
Inércia de spin não-Markoviana em canais 2D (grafeno / WSe₂).
"""

import numpy as np
from typing import Optional, Tuple, Deque
from collections import deque


class DinamicaSpinRetardada:
    """
    Estado de spin escalar M_t ∈ [-1, 1] (polarização):

        dM_t = [ −γ M_t + α_STT u_t − β M_{t−τ} ] dt + σ dW_t

    O termo β M_{t−τ} codifica a inércia / memória de spin
    com tempo de vida τ = τ_spin.
    """

    def __init__(
        self,
        tau: float = 0.15,         # retardo de spin
        gamma: float = 0.8,        # relaxação Markoviana
        beta_mem: float = 0.35,    # força da memória
        alpha_stt: float = 1.0,    # acoplamento STT ao controle u
        sigma: float = 0.08,
        M0: float = 0.0,
        dt: float = 0.01,
        semente: Optional[int] = 42,
    ):
        self.tau = tau
        self.gamma = gamma
        self.beta_mem = beta_mem
        self.alpha_stt = alpha_stt
        self.sigma = sigma
        self.dt = dt
        self.rng = np.random.default_rng(semente)

        self.n_delay = max(1, int(round(tau / dt)))
        self.historico: Deque[float] = deque([M0] * (self.n_delay + 1), maxlen=self.n_delay + 1)
        self.M = float(M0)
        self.t = 0.0

    def M_retardado(self) -> float:
        return float(self.historico[0])  # mais antigo ≈ t−τ

    def passo(self, u: float = 0.0) -> Tuple[float, float]:
        """Um passo Euler–Maruyama; retorna (M_t, M_{t−τ})."""
        M_tau = self.M_retardado()
        drift = -self.gamma * self.M + self.alpha_stt * u - self.beta_mem * M_tau
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.M = float(np.clip(self.M + drift * self.dt + self.sigma * dW, -1.0, 1.0))
        self.historico.append(self.M)
        self.t += self.dt
        return self.M, M_tau

    def simular(self, n_passos: int, politica_u) -> dict:
        """
        politica_u(M, M_tau, t) → u
        """
        traj_M = np.zeros(n_passos)
        traj_Mtau = np.zeros(n_passos)
        traj_u = np.zeros(n_passos)
        traj_t = np.zeros(n_passos)
        for k in range(n_passos):
            M_tau = self.M_retardado()
            u = float(politica_u(self.M, M_tau, self.t))
            traj_t[k] = self.t
            traj_M[k] = self.M
            traj_Mtau[k] = M_tau
            traj_u[k] = u
            self.passo(u)
        return {"t": traj_t, "M": traj_M, "M_tau": traj_Mtau, "u": traj_u}
