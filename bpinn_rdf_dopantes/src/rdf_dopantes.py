"""
Flutuações aleatórias de dopantes (RDF) em canal ~1.6 nm.
"""

import numpy as np
from typing import Optional, Tuple


class CanalRDF:
    """
    Canal 1D efetivo (corte longitudinal) com dopantes em posições aleatórias.
    Cada dopante contribui com carga puntual / Gaussiana estreita ao ρ(x).
    """

    def __init__(
        self,
        L: float = 1.0,              # ~ comprimento normalizado (1.6 nm se unidade = 1.6 nm)
        n_dopantes: int = 8,
        carga_dopante: float = 1.0,
        largura_gauss: float = 0.04,
        V_source: float = 0.0,
        V_drain: float = 0.3,
        semente: Optional[int] = 42,
    ):
        self.L = L
        self.n_dopantes = n_dopantes
        self.carga_dopante = carga_dopante
        self.largura_gauss = largura_gauss
        self.V_source = V_source
        self.V_drain = V_drain
        self.rng = np.random.default_rng(semente)
        self.posicoes = self.rng.uniform(0.1 * L, 0.9 * L, n_dopantes)

    def densidade_carga(self, x: np.ndarray) -> np.ndarray:
        rho = np.zeros_like(x, dtype=float)
        for p in self.posicoes:
            rho += self.carga_dopante * np.exp(-0.5 * ((x - p) / self.largura_gauss) ** 2)
        return rho

    def potencial_referencia(self, x: np.ndarray, epsilon: float = 1.0) -> np.ndarray:
        """
        Solução 1D aproximada de Poisson −ε φ'' = ρ
        com Dirichlet nas bordas (diferenças finitas).
        """
        n = len(x)
        dx = x[1] - x[0]
        A = np.zeros((n, n))
        b = np.zeros(n)
        A[0, 0] = 1.0
        b[0] = self.V_source
        A[-1, -1] = 1.0
        b[-1] = self.V_drain
        coef = epsilon / (dx ** 2)
        rho = self.densidade_carga(x)
        for i in range(1, n - 1):
            A[i, i - 1] = coef
            A[i, i] = -2.0 * coef
            A[i, i + 1] = coef
            b[i] = -rho[i]
        return np.linalg.solve(A, b)

    def nova_realizacao(self, semente: Optional[int] = None) -> "CanalRDF":
        return CanalRDF(
            self.L, self.n_dopantes, self.carga_dopante, self.largura_gauss,
            self.V_source, self.V_drain,
            semente if semente is not None else int(self.rng.integers(0, 1e9)),
        )
