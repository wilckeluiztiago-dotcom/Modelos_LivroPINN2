"""
Interpretação física TAT em HfO₂/ZrO₂:
S ~ energia/posição efetiva do portador na armadilha;
saltos = tunelamento entre defeitos; difusão = Poole–Frenkel térmico.
"""

import numpy as np
from typing import Optional


class DieletricoTAT:
    """
    Parâmetros efetivos do dielétrico de porta ultra-fino.
    """

    def __init__(
        self,
        espessura: float = 1.6e-9,   # 1.6 nm
        lambda_salto: float = 1.2,  # taxa de tunelamento entre armadilhas
        sigma_termico: float = 0.3, # agitação Poole–Frenkel
        r_relax: float = 0.1,       # taxa de relaxação / captura
        mu_j: float = -0.15,
        sig_j: float = 0.35,
    ):
        self.espessura = espessura
        self.lambda_salto = lambda_salto
        self.sigma_termico = sigma_termico
        self.r_relax = r_relax
        self.mu_j = mu_j
        self.sig_j = sig_j

    def kappa(self) -> float:
        return float(np.exp(self.mu_j + 0.5 * self.sig_j ** 2) - 1.0)
