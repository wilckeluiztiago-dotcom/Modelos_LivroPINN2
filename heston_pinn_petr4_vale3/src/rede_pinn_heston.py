"""
PINN V(S, v, τ) para a EDP de Heston.
"""

import numpy as np
from typing import List, Optional, Dict


class RedePINN_Heston:
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
        return np.log1p(np.exp(np.clip(a.squeeze(), -20, 20)))

    def prever(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def derivadas(self, X: np.ndarray, eps: float = 1e-4):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        V = self.forward(X)

        def d1(j):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps; Xm[:, j] -= eps
            return (self.forward(Xp) - self.forward(Xm)) / (2 * eps)

        def d2(j):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps; Xm[:, j] -= eps
            return (self.forward(Xp) - 2 * V + self.forward(Xm)) / (eps ** 2)

        VS, Vv, Vtau = d1(0), d1(1), d1(2)
        VSS, Vvv = d2(0), d2(1)
        Xp = X.copy(); Xm = X.copy()
        Xp[:, 0] += eps; Xp[:, 1] += eps
        Xm[:, 0] -= eps; Xm[:, 1] -= eps
        Xp2 = X.copy(); Xm2 = X.copy()
        Xp2[:, 0] += eps; Xp2[:, 1] -= eps
        Xm2[:, 0] -= eps; Xm2[:, 1] += eps
        VSv = (self.forward(Xp) - self.forward(Xp2) - self.forward(Xm2) + self.forward(Xm)) / (4 * eps ** 2)
        return V, VS, Vv, Vtau, VSS, VSv, Vvv

    def gregas(self, X: np.ndarray, eps: float = 1e-4) -> Dict:
        V, VS, Vv, Vtau, VSS, VSv, Vvv = self.derivadas(X, eps)
        return {
            "V": V,
            "Delta": VS,
            "Gamma": VSS,
            "Vanna": VSv,
            "Vega_v": Vv,
        }

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
