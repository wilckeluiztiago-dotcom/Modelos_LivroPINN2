"""
PINN para mapa de condutância / oscilações de Coulomb
em função da polarização de gate (sem malha espacial).
"""

import numpy as np
from typing import List, Optional, Dict, Tuple


class RedePINN:
    def __init__(self, camadas: List[int] = None, semente: Optional[int] = 42):
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

    def prever(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def gradiente(self, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        x = np.atleast_1d(x).astype(float)
        return (self.forward(x + eps) - self.forward(x - eps)) / (2 * eps)

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


def perda_condutancia(
    rede: RedePINN,
    V_gate: np.ndarray,
    G_dados: np.ndarray,
    peso_dados: float = 1.0,
    peso_suave: float = 0.05,
) -> float:
    """
    Ajuste aos dados de condutância + regularização de suavidade
    (PINN fraca: dados de transporte + prior suave em ∂G/∂V).
    """
    pred = rede.prever(V_gate)
    perda_d = float(np.mean((pred - G_dados) ** 2))
    # suavidade
    g = rede.gradiente(V_gate)
    perda_s = float(np.mean(g ** 2))
    return peso_dados * perda_d + peso_suave * perda_s


def treinar_condutancia(
    rede: RedePINN,
    V_gate: np.ndarray,
    G_dados: np.ndarray,
    n_epocas: int = 400,
    taxa: float = 1e-3,
    semente: Optional[int] = 0,
    verbose_cada: int = 50,
) -> Dict:
    g = np.random.default_rng(semente)
    theta = rede.parametros_vetor().copy()
    n_params = len(theta)
    historico = []
    melhor = np.inf
    melhor_theta = theta.copy()
    m = np.zeros_like(theta)
    eps_g = 1e-5

    for epoca in range(1, n_epocas + 1):
        p0 = perda_condutancia(rede, V_gate, G_dados)
        grad = np.zeros_like(theta)
        idx = g.choice(n_params, size=min(28, n_params), replace=False)
        for j in idx:
            tp = theta.copy()
            tp[j] += eps_g
            rede.carregar_parametros(tp)
            grad[j] = (perda_condutancia(rede, V_gate, G_dados) - p0) / eps_g
        m = 0.9 * m + 0.1 * grad
        theta = theta - taxa * m
        rede.carregar_parametros(theta)
        perda = perda_condutancia(rede, V_gate, G_dados)
        historico.append(perda)
        if perda < melhor:
            melhor = perda
            melhor_theta = theta.copy()
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:4d} | perda={perda:.4e}")
        if epoca % 150 == 0:
            taxa *= 0.8

    rede.carregar_parametros(melhor_theta)
    return {"historico": historico, "perda_final": melhor}
