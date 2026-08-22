"""
PINN de dois campos: T_e(x,t) e T_L(x,t).
Uma rede com saída 2, ou duas redes; aqui saída 2.
"""

import numpy as np
from typing import List, Optional, Tuple


class RedePINN_TTM:
    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
        if camadas is None:
            camadas = [2, 40, 40, 2]  # (x,t) → (Te, TL)
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
        # temperaturas positivas
        return np.log1p(np.exp(np.clip(a, -20, 20)))

    def prever(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        return out.squeeze()

    def campos(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        out = self.prever(X)
        if out.ndim == 1:
            return out[0], out[1]
        return out[:, 0], out[:, 1]

    def derivadas(self, X: np.ndarray, eps: float = 1e-4):
        """∂_t, ∂_x, ∂_xx de Te e TL."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        Te, TL = self.campos(X)

        def d(campo_idx, coord_idx):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, coord_idx] += eps; Xm[:, coord_idx] -= eps
            op = self.prever(Xp); om = self.prever(Xm)
            if op.ndim == 1:
                return (op[campo_idx] - om[campo_idx]) / (2 * eps)
            return (op[:, campo_idx] - om[:, campo_idx]) / (2 * eps)

        def d2(campo_idx, coord_idx):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, coord_idx] += eps; Xm[:, coord_idx] -= eps
            op = self.prever(Xp); om = self.prever(Xm)
            o0 = self.prever(X)
            if op.ndim == 1:
                return (op[campo_idx] - 2 * o0[campo_idx] + om[campo_idx]) / (eps ** 2)
            return (op[:, campo_idx] - 2 * o0[:, campo_idx] + om[:, campo_idx]) / (eps ** 2)

        # coord 0 = x, coord 1 = t
        Te_x, Te_t = d(0, 0), d(0, 1)
        TL_x, TL_t = d(1, 0), d(1, 1)
        Te_xx, TL_xx = d2(0, 0), d2(1, 0)
        return Te, TL, Te_t, TL_t, Te_xx, TL_xx

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
