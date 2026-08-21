"""
PI-DeepONet: Branch (curva DI) + Trunk (t, T) → P(t,T).
"""

import numpy as np
from typing import List, Optional, Tuple


def _mlp_init(camadas, semente):
    g = np.random.default_rng(semente)
    pesos, vieses = [], []
    for i in range(len(camadas) - 1):
        lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
        pesos.append(g.uniform(-lim, lim, (camadas[i], camadas[i + 1])))
        vieses.append(np.zeros(camadas[i + 1]))
    return pesos, vieses


def _mlp_forward(x, pesos, vieses):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    a = x
    for i, (W, b) in enumerate(zip(pesos, vieses)):
        z = a @ W + b
        a = np.tanh(z) if i < len(pesos) - 1 else z
    return a


class BranchNet:
    """Processa vértices da curva DI → embedding."""

    def __init__(self, n_vertices: int = 8, dim_out: int = 32, semente: int = 42):
        self.n_vertices = n_vertices
        self.dim_out = dim_out
        self.pesos, self.vieses = _mlp_init([n_vertices, 64, 64, dim_out], semente)

    def forward(self, curva: np.ndarray) -> np.ndarray:
        """curva: (batch, n_vertices) → (batch, dim_out)"""
        return _mlp_forward(curva, self.pesos, self.vieses)

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes += [W.ravel(), b.ravel()]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray, idx: int = 0) -> int:
        for i in range(len(self.pesos)):
            nW = self.pesos[i].size
            self.pesos[i] = theta[idx:idx + nW].reshape(self.pesos[i].shape)
            idx += nW
            nb = self.vieses[i].size
            self.vieses[i] = theta[idx:idx + nb]
            idx += nb
        return idx


class TrunkNet:
    """Processa (t, T) → embedding."""

    def __init__(self, dim_out: int = 32, semente: int = 43):
        self.dim_out = dim_out
        self.pesos, self.vieses = _mlp_init([2, 64, 64, dim_out], semente)

    def forward(self, tT: np.ndarray) -> np.ndarray:
        return _mlp_forward(tT, self.pesos, self.vieses)

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes += [W.ravel(), b.ravel()]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray, idx: int = 0) -> int:
        for i in range(len(self.pesos)):
            nW = self.pesos[i].size
            self.pesos[i] = theta[idx:idx + nW].reshape(self.pesos[i].shape)
            idx += nW
            nb = self.vieses[i].size
            self.vieses[i] = theta[idx:idx + nb]
            idx += nb
        return idx


class PIDeepONet:
    """
    P_θ(curva; t, T) = Branch(curva) · Trunk(t,T) + bias
    """

    def __init__(self, n_vertices: int = 8, dim: int = 32, semente: int = 42):
        self.branch = BranchNet(n_vertices, dim, semente)
        self.trunk = TrunkNet(dim, semente + 1)
        self.bias = np.array([0.5])
        self.dim = dim
        self.n_vertices = n_vertices

    def prever(self, curva: np.ndarray, tT: np.ndarray) -> np.ndarray:
        """
        curva: (n_vertices,) ou (1, n_vertices)
        tT: (N, 2)
        → P: (N,)
        """
        b = self.branch.forward(curva.reshape(1, -1))  # (1, dim)
        tr = self.trunk.forward(tT)  # (N, dim)
        out = (tr * b).sum(axis=1) + self.bias[0]
        # preços em (0, 1.5]
        return 1.0 / (1.0 + np.exp(-np.clip(out, -20, 20)))  # sigmoid ~ (0,1)

    def parametros_vetor(self) -> np.ndarray:
        return np.concatenate([
            self.branch.parametros_vetor(),
            self.trunk.parametros_vetor(),
            self.bias,
        ])

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = self.branch.carregar_parametros(theta, 0)
        idx = self.trunk.carregar_parametros(theta, idx)
        self.bias = theta[idx:idx + 1].copy()

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())
