"""
Célula neural do Deep Galerkin Method (DGM).
Arquitetura recorrente residual estilo LSTM (Sirignano & Spiliopoulos).

Referência estrutural alinhada ao formalismo de redes profundas
para EDPs de alta dimensão.
"""

import numpy as np
from typing import List, Optional, Tuple


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class CelulaDGM:
    """
    Uma camada DGM:

        Z = σ(W_z · [x, S] + b_z)
        G = σ(W_g · [x, S] + b_g)
        R = σ(W_r · [x, S] + b_r)
        H = tanh(W_h · [x, R ⊙ S] + b_h)
        S_new = (1 - G) ⊙ H + Z ⊙ S

    onde S é o estado oculto e x a entrada (espaço de fases + tempo).
    """

    def __init__(self, dim_in: int, dim_hidden: int, semente: Optional[int] = None):
        g = np.random.default_rng(semente)
        lim = np.sqrt(6.0 / (dim_in + dim_hidden))
        self.Wz = g.uniform(-lim, lim, (dim_in + dim_hidden, dim_hidden))
        self.bz = np.zeros(dim_hidden)
        self.Wg = g.uniform(-lim, lim, (dim_in + dim_hidden, dim_hidden))
        self.bg = np.zeros(dim_hidden)
        self.Wr = g.uniform(-lim, lim, (dim_in + dim_hidden, dim_hidden))
        self.br = np.zeros(dim_hidden)
        self.Wh = g.uniform(-lim, lim, (dim_in + dim_hidden, dim_hidden))
        self.bh = np.zeros(dim_hidden)
        self.dim_hidden = dim_hidden
        self.dim_in = dim_in

    def forward(self, x: np.ndarray, S: np.ndarray) -> np.ndarray:
        """x: (N, dim_in), S: (N, dim_hidden) → S_new."""
        xs = np.concatenate([x, S], axis=1)
        Z = sigmoid(xs @ self.Wz + self.bz)
        G = sigmoid(xs @ self.Wg + self.bg)
        R = sigmoid(xs @ self.Wr + self.br)
        xrs = np.concatenate([x, R * S], axis=1)
        H = np.tanh(xrs @ self.Wh + self.bh)
        return (1.0 - G) * H + Z * S

    def parametros(self) -> List[np.ndarray]:
        return [self.Wz, self.bz, self.Wg, self.bg, self.Wr, self.br, self.Wh, self.bh]

    def carregar(self, params: List[np.ndarray]) -> None:
        self.Wz, self.bz, self.Wg, self.bg, self.Wr, self.br, self.Wh, self.bh = params


class RedeDGM:
    """
    Rede DGM completa: camada de entrada → L células DGM → saída escalar f_W.
    """

    def __init__(
        self,
        dim_entrada: int = 3,      # (x, kx, t) na demo; 7 para (x,y,z,kx,ky,kz,t)
        dim_oculta: int = 64,
        n_camadas: int = 3,
        semente: Optional[int] = 42,
    ):
        self.dim_entrada = dim_entrada
        self.dim_oculta = dim_oculta
        self.n_camadas = n_camadas
        g = np.random.default_rng(semente)

        lim0 = np.sqrt(6.0 / (dim_entrada + dim_oculta))
        self.W0 = g.uniform(-lim0, lim0, (dim_entrada, dim_oculta))
        self.b0 = np.zeros(dim_oculta)

        self.celulas = [
            CelulaDGM(dim_entrada, dim_oculta, semente=semente + i + 1)
            for i in range(n_camadas)
        ]

        lim_out = np.sqrt(6.0 / (dim_oculta + 1))
        self.Wout = g.uniform(-lim_out, lim_out, (dim_oculta, 1))
        self.bout = np.zeros(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """X: (N, dim_entrada) → f_W: (N,)"""
        if X.ndim == 1:
            X = X.reshape(1, -1)
        S = np.tanh(X @ self.W0 + self.b0)
        for cel in self.celulas:
            S = cel.forward(X, S)
        return (S @ self.Wout + self.bout).squeeze()

    def prever(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def gradiente(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """∇_{X} f_W, shape (N, dim)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        N, d = X.shape
        g = np.zeros((N, d))
        for j in range(d):
            Xp = X.copy(); Xm = X.copy()
            Xp[:, j] += eps
            Xm[:, j] -= eps
            g[:, j] = (self.forward(Xp) - self.forward(Xm)) / (2 * eps)
        return g

    def derivada_temporal(self, X: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """∂f/∂t — assume última coluna é t."""
        return self.gradiente(X, eps)[:, -1]

    def parametros_vetor(self) -> np.ndarray:
        partes = [self.W0.ravel(), self.b0.ravel()]
        for cel in self.celulas:
            for p in cel.parametros():
                partes.append(p.ravel())
        partes.append(self.Wout.ravel())
        partes.append(self.bout.ravel())
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = 0

        def take(shape):
            nonlocal idx
            n = int(np.prod(shape))
            arr = theta[idx:idx + n].reshape(shape)
            idx += n
            return arr

        self.W0 = take(self.W0.shape)
        self.b0 = take(self.b0.shape)
        for cel in self.celulas:
            params = [take(p.shape) for p in cel.parametros()]
            cel.carregar(params)
        self.Wout = take(self.Wout.shape)
        self.bout = take(self.bout.shape)

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())
