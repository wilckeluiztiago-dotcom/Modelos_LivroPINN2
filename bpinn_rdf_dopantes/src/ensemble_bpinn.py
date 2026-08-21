"""
Ensemble de PINNs como aproximação Bayesiana prática (Apêndice C.3).
Cada membro é treinado com semente distinta → incerteza epistêmica.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


class MembroPINN:
    def __init__(self, camadas: List[int] = None, semente: int = 0):
        if camadas is None:
            camadas = [1, 32, 32, 1]
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

    def laplaciano(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
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


def perda_membro(rede: MembroPINN, x_col, rho, x_bc, v_bc, epsilon=1.0, peso_bc=12.0) -> float:
    lap = rede.laplaciano(x_col)
    res = -epsilon * lap - rho
    pde = float(np.mean(res ** 2))
    bc = float(np.mean((rede.forward(x_bc) - v_bc) ** 2))
    return pde + peso_bc * bc


def treinar_membro(
    rede: MembroPINN,
    x_col, rho, x_bc, v_bc,
    n_epocas: int = 250,
    taxa: float = 8e-4,
    semente: int = 0,
) -> List[float]:
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    hist = []
    m = np.zeros_like(theta)
    eps_g = 1e-5
    melhor = np.inf
    melhor_theta = theta.copy()
    for epoca in range(1, n_epocas + 1):
        p0 = perda_membro(rede, x_col, rho, x_bc, v_bc)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(32, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            grad[j] = (perda_membro(rede, x_col, rho, x_bc, v_bc) - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda = perda_membro(rede, x_col, rho, x_bc, v_bc)
        hist.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if epoca % 100 == 0:
            taxa *= 0.8
    rede.carregar_parametros(melhor_theta)
    return hist


class EnsembleBPINN:
    """Ensemble de M redes = aproximação do posterior preditivo."""

    def __init__(self, n_membros: int = 5, camadas=None, semente: int = 42):
        self.membros = [
            MembroPINN(camadas=camadas, semente=semente + k * 17)
            for k in range(n_membros)
        ]
        self.historicos = []

    def treinar(self, x_col, rho, x_bc, v_bc, n_epocas=250, verbose=True):
        for k, m in enumerate(self.membros):
            if verbose:
                print(f"  membro {k+1}/{len(self.membros)}...")
            h = treinar_membro(m, x_col, rho, x_bc, v_bc, n_epocas=n_epocas, semente=k)
            self.historicos.append(h)
            if verbose:
                print(f"    perda final={h[-1]:.4e}")

    def prever_media_var(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        preds = np.stack([m.forward(x) for m in self.membros], axis=0)
        return preds.mean(axis=0), preds.var(axis=0)
