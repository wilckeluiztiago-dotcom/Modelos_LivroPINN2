"""
PINN para p(q, s, t) — q tratado como canal one-hot / embutido contínuo.
"""

import numpy as np
from typing import List, Optional


class RedePINN_SET:
    """
    Entrada (q_norm, s, t) → p ≥ 0.
    q_norm = q / q_max (tratado continuamente para autograd por diferenças).
    """

    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
        if camadas is None:
            camadas = [3, 32, 32, 1]
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

    def derivadas_s_t(self, X: np.ndarray, eps: float = 1e-4):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        p = self.forward(X)
        # ∂/∂s (coluna 1)
        Xp = X.copy(); Xm = X.copy()
        Xp[:, 1] += eps; Xm[:, 1] -= eps
        ps = (self.forward(Xp) - self.forward(Xm)) / (2 * eps)
        pss = (self.forward(Xp) - 2 * p + self.forward(Xm)) / (eps ** 2)
        # ∂/∂t (coluna 2)
        Xtp = X.copy(); Xtm = X.copy()
        Xtp[:, 2] += eps; Xtm[:, 2] -= eps
        pt = (self.forward(Xtp) - self.forward(Xtm)) / (2 * eps)
        return p, pt, ps, pss

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
