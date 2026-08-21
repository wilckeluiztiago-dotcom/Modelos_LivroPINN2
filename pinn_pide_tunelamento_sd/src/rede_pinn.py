"""Rede PINN 1D (posição → densidade / potencial de corrente)."""
import numpy as np
from typing import List, Optional


class RedePINN:
    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
        if camadas is None:
            camadas = [1, 32, 32, 32, 1]
        self.camadas = camadas
        self.n_camadas = len(camadas) - 1
        g = np.random.default_rng(semente)
        self.pesos, self.vieses = [], []
        for i in range(self.n_camadas):
            lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
            self.pesos.append(g.uniform(-lim, lim, (camadas[i], camadas[i + 1])))
            self.vieses.append(np.zeros(camadas[i + 1]))

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        a = x
        for i in range(self.n_camadas):
            z = a @ self.pesos[i] + self.vieses[i]
            a = np.tanh(z) if i < self.n_camadas - 1 else z
        return a.squeeze()

    def prever(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def gradiente(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        x = np.atleast_1d(x).astype(float)
        return (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)

    def hessiana(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        x = np.atleast_1d(x).astype(float)
        return (self.forward(x + eps) - 2 * self.forward(x) + self.forward(x - eps)) / (eps ** 2)

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
