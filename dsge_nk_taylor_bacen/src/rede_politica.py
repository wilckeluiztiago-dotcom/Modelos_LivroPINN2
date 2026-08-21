"""
Rede neural de função de política:
  entrada (r̂^n, tt, fisc) → (ŷ, π̂, î)
"""

import numpy as np
from typing import List, Optional


class RedePoliticaNK:
    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
        if camadas is None:
            camadas = [3, 32, 32, 3]
        self.camadas = camadas
        self.n_camadas = len(camadas) - 1
        g = np.random.default_rng(semente)
        self.pesos, self.vieses = [], []
        for i in range(self.n_camadas):
            lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
            self.pesos.append(g.uniform(-lim, lim, (camadas[i], camadas[i + 1])))
            self.vieses.append(np.zeros(camadas[i + 1]))

    def forward(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        a = X
        for i in range(self.n_camadas):
            z = a @ self.pesos[i] + self.vieses[i]
            a = np.tanh(z) if i < self.n_camadas - 1 else z
        return a  # (batch, 3) = (y, pi, i)

    def prever(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return out.squeeze()

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes += [W.ravel(), b.ravel()]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = 0
        for i in range(self.n_camadas):
            nW = self.pesos[i].size
            self.pesos[i] = theta[idx:idx + nW].reshape(self.pesos[i].shape)
            idx += nW
            nb = self.vieses[i].size
            self.vieses[i] = theta[idx:idx + nb]
            idx += nb

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())
