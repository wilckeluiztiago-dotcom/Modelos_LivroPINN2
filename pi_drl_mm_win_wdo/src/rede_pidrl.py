"""
Redes Actor e Critic para PI-DRL de market making.
"""

import numpy as np
from typing import List, Optional, Tuple


def _init(camadas, semente):
    g = np.random.default_rng(semente)
    pesos, vieses = [], []
    for i in range(len(camadas) - 1):
        lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
        pesos.append(g.uniform(-lim, lim, (camadas[i], camadas[i + 1])))
        vieses.append(np.zeros(camadas[i + 1]))
    return pesos, vieses


def _fwd(x, pesos, vieses, ativ_final=None):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    a = x
    for i, (W, b) in enumerate(zip(pesos, vieses)):
        z = a @ W + b
        if i < len(pesos) - 1:
            a = np.tanh(z)
        else:
            a = z if ativ_final is None else ativ_final(z)
    return a.squeeze()


class Critic:
    """V(t, s, q) — valor da carteira / função valor HJB."""

    def __init__(self, semente: int = 42):
        self.pesos, self.vieses = _init([3, 32, 32, 1], semente)

    def valor(self, estado: np.ndarray) -> np.ndarray:
        """estado: (t, s_norm, q_norm) ou batch."""
        return _fwd(estado, self.pesos, self.vieses)

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes += [W.ravel(), b.ravel()]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = 0
        for i in range(len(self.pesos)):
            nW = self.pesos[i].size
            self.pesos[i] = theta[idx:idx + nW].reshape(self.pesos[i].shape)
            idx += nW
            nb = self.vieses[i].size
            self.vieses[i] = theta[idx:idx + nb]
            idx += nb

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())


class Actor:
    """Política: estado → (δ^b, δ^a) spreads."""

    def __init__(self, semente: int = 43):
        self.pesos, self.vieses = _init([3, 32, 32, 2], semente)

    def spreads(self, estado: np.ndarray) -> np.ndarray:
        out = _fwd(estado, self.pesos, self.vieses)
        # softplus para δ > 0
        if out.ndim == 1:
            return np.log1p(np.exp(np.clip(out, -20, 20)))
        return np.log1p(np.exp(np.clip(out, -20, 20)))

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for W, b in zip(self.pesos, self.vieses):
            partes += [W.ravel(), b.ravel()]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = 0
        for i in range(len(self.pesos)):
            nW = self.pesos[i].size
            self.pesos[i] = theta[idx:idx + nW].reshape(self.pesos[i].shape)
            idx += nW
            nb = self.vieses[i].size
            self.vieses[i] = theta[idx:idx + nb]
            idx += nb

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())
