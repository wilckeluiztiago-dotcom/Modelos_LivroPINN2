"""
Rede neural PINN 3D (entrada x,y,z → potencial φ).
Base: Cap. 2 do livro — Teorema de Aproximação Universal + autograd conceitual.
"""

import numpy as np
from typing import List, Optional


class RedePINN3D:
    """
    MLP com entrada 3D (x, y, z) e saída escalar φ_θ(x,y,z).

    Diferenciação por diferenças finitas centradas (equivalente didático
    ao autograd do Cap. 2.4), suficiente para o resíduo de Poisson.
    """

    def __init__(
        self,
        camadas: List[int] = None,
        semente: Optional[int] = 42,
    ):
        if camadas is None:
            camadas = [3, 48, 48, 48, 1]
        self.camadas = camadas
        self.n_camadas = len(camadas) - 1
        gerador = np.random.default_rng(semente)
        self.pesos: List[np.ndarray] = []
        self.vieses: List[np.ndarray] = []
        for i in range(self.n_camadas):
            lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
            W = gerador.uniform(-lim, lim, size=(camadas[i], camadas[i + 1]))
            b = np.zeros(camadas[i + 1])
            self.pesos.append(W)
            self.vieses.append(b)

    def _ativacao(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        X: (N, 3) → φ: (N,)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
        a = X
        for i in range(self.n_camadas):
            z = a @ self.pesos[i] + self.vieses[i]
            a = self._ativacao(z) if i < self.n_camadas - 1 else z
        return a.squeeze()

    def prever(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def gradiente(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """∇φ = (∂φ/∂x, ∂φ/∂y, ∂φ/∂z), shape (N, 3)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        g = np.zeros((N, 3))
        for d in range(3):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, d] += eps
            Xm[:, d] -= eps
            g[:, d] = (self.forward(Xp) - self.forward(Xm)) / (2.0 * eps)
        return g

    def laplaciano(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Δφ = ∂²φ/∂x² + ∂²φ/∂y² + ∂²φ/∂z²."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        lap = np.zeros(N)
        phi0 = self.forward(X)
        for d in range(3):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, d] += eps
            Xm[:, d] -= eps
            lap += (self.forward(Xp) - 2.0 * phi0 + self.forward(Xm)) / (eps ** 2)
        return lap

    def divergencia_epsilon_grad(
        self,
        X: np.ndarray,
        epsilon_fn,
        eps: float = 1e-4,
    ) -> np.ndarray:
        """
        ∇ · (ε ∇φ) por diferenças finitas.
        epsilon_fn(X) → ε em cada ponto.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N = X.shape[0]
        div = np.zeros(N)
        # ∇φ
        grad = self.gradiente(X, eps)
        eps_c = epsilon_fn(X[:, 0], X[:, 1], X[:, 2])
        flux = eps_c[:, None] * grad  # (N, 3)
        # divergência do fluxo
        for d in range(3):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, d] += eps
            Xm[:, d] -= eps
            gp = self.gradiente(Xp, eps)
            gm = self.gradiente(Xm, eps)
            ep = epsilon_fn(Xp[:, 0], Xp[:, 1], Xp[:, 2])
            em = epsilon_fn(Xm[:, 0], Xm[:, 1], Xm[:, 2])
            fp = ep * gp[:, d]
            fm = em * gm[:, d]
            div += (fp - fm) / (2.0 * eps)
        return div

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes.append(W.ravel())
            partes.append(b.ravel())
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
        return sum(W.size + b.size for W, b in zip(self.pesos, self.vieses))
