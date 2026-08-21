"""
PINN 3D: entrada (r, i, τ) → P_nom e/ou P_real.
"""

import numpy as np
from typing import List, Optional


class RedePINN_G2:
    """Entrada (r, i, τ) → preço (nominal ou real conforme treino)."""

    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
        if camadas is None:
            camadas = [3, 40, 40, 1]
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
        # preço em (0, 1.2]
        return 1.0 / (1.0 + np.exp(-np.clip(a.squeeze(), -20, 20)))

    def prever(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def derivadas(self, X: np.ndarray, eps: float = 1e-4):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        P = self.forward(X)
        grads = []
        for j in range(3):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps; Xm[:, j] -= eps
            grads.append((self.forward(Xp) - self.forward(Xm)) / (2 * eps))
        # Hessiana diagonal aproximada em r e i
        Xpr = X.copy(); Xmr = X.copy()
        Xpr[:, 0] += eps; Xmr[:, 0] -= eps
        Prr = (self.forward(Xpr) - 2 * P + self.forward(Xmr)) / (eps ** 2)
        Xpi = X.copy(); Xmi = X.copy()
        Xpi[:, 1] += eps; Xmi[:, 1] -= eps
        Pii = (self.forward(Xpi) - 2 * P + self.forward(Xmi)) / (eps ** 2)
        return P, grads[0], grads[1], grads[2], Prr, Pii

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
