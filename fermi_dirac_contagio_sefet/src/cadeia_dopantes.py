"""
Cadeia de dopantes discretos em nanofio ultra-estreito.
Energias de sítio D_i e interação de Coulomb.
"""

import numpy as np
from typing import Optional, Tuple


class CadeiaDopantes:
    """
    N sítios de dopantes ao longo do canal (Single-Electron FET).

    Cada sítio i tem energia de sítio D_i e ocupação n_i ∈ {0,1}
    (bloqueio de Coulomb: no máximo um elétron por sítio em regime SE).
    """

    def __init__(
        self,
        n_sitios: int = 12,
        L: float = 1.0,
        U_coulomb: float = 1.5,      # energia de carregamento
        desordem: float = 0.3,       # desordem de sítio
        V_source: float = 0.0,
        V_drain: float = 0.4,
        semente: Optional[int] = 42,
    ):
        self.n_sitios = n_sitios
        self.L = L
        self.U_coulomb = U_coulomb
        self.desordem = desordem
        self.V_source = V_source
        self.V_drain = V_drain
        g = np.random.default_rng(semente)

        self.x = np.linspace(0, L, n_sitios)
        # energias de sítio com desordem + rampa source–drain
        self.D0 = g.normal(0.0, desordem, n_sitios)
        self.D0 += V_source + (V_drain - V_source) * (self.x / L)

        self.ocupacao = np.zeros(n_sitios, dtype=int)  # 0 ou 1

    def energia_sitio(self, i: int) -> float:
        """
        Energia efetiva do sítio i incluindo Coulomb dos vizinhos ocupados:
            D_i^{eff} = D_i^0 + U Σ_{j≠i} n_j / (1 + |x_i-x_j|/λ)
        """
        U_sum = 0.0
        for j in range(self.n_sitios):
            if j != i and self.ocupacao[j] == 1:
                dist = abs(self.x[i] - self.x[j]) + 0.15
                U_sum += self.U_coulomb / dist
        return float(self.D0[i] + U_sum)

    def energias_efetivas(self) -> np.ndarray:
        return np.array([self.energia_sitio(i) for i in range(self.n_sitios)])

    def numero_eletrons(self) -> int:
        return int(np.sum(self.ocupacao))
