"""
Gerador e Discriminador para I-GAN de mapas WFF.
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


def _mlp_forward(x, pesos, vieses, ultima_ativacao=None):
    if x.ndim == 1:
        x = x.reshape(1, -1)
    a = x
    for i, (W, b) in enumerate(zip(pesos, vieses)):
        z = a @ W + b
        if i < len(pesos) - 1:
            a = np.tanh(z)
        else:
            if ultima_ativacao == "sigmoid":
                a = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            elif ultima_ativacao == "tanh":
                a = np.tanh(z)
            else:
                a = z
    return a.squeeze()


class Gerador:
    """z ~ N(0,I) → mapa WF achatado (nx*ny)."""

    def __init__(self, dim_z: int = 16, nx: int = 32, ny: int = 16, semente: int = 42):
        self.dim_z = dim_z
        self.nx, self.ny = nx, ny
        self.dim_out = nx * ny
        camadas = [dim_z, 64, 64, self.dim_out]
        self.pesos, self.vieses = _mlp_init(camadas, semente)
        self.camadas = camadas

    def gerar(self, z: np.ndarray) -> np.ndarray:
        out = _mlp_forward(z, self.pesos, self.vieses, ultima_ativacao="tanh")
        # escala para faixa de WF ~ [0.7, 1.3]
        if out.ndim == 1:
            out = out.reshape(1, -1)
        wf = 1.0 + 0.3 * out
        return wf.reshape(-1, self.nx, self.ny)

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


class Discriminador:
    """mapa WF achatado → probabilidade real/fake."""

    def __init__(self, nx: int = 32, ny: int = 16, semente: int = 43):
        self.nx, self.ny = nx, ny
        camadas = [nx * ny, 64, 32, 1]
        self.pesos, self.vieses = _mlp_init(camadas, semente)
        self.camadas = camadas

    def discriminar(self, wf: np.ndarray) -> np.ndarray:
        if wf.ndim == 3:
            wf = wf.reshape(wf.shape[0], -1)
        elif wf.ndim == 2 and wf.shape == (self.nx, self.ny):
            wf = wf.reshape(1, -1)
        return _mlp_forward(wf, self.pesos, self.vieses, ultima_ativacao="sigmoid")

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
