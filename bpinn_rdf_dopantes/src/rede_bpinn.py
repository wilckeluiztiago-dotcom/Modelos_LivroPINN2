"""
Rede com pesos estocásticos (aproximação Bayesiana / ensemble).
Apêndice C.3 — prior Gaussiano sobre θ.
"""

import numpy as np
from typing import List, Optional, Tuple


class RedeBayesiana:
    """
    MLP com amostragem de pesos a partir de
        θ ~ N(μ_θ, σ_θ² I)   (posterior variacional diagonal)
    """

    def __init__(
        self,
        camadas: List[int] = None,
        prior_std: float = 0.1,
        semente: Optional[int] = 42,
    ):
        if camadas is None:
            camadas = [1, 32, 32, 1]
        self.camadas = camadas
        self.n_camadas = len(camadas) - 1
        self.prior_std = prior_std
        g = np.random.default_rng(semente)

        self.mu_pesos, self.mu_vieses = [], []
        self.log_sigma_pesos, self.log_sigma_vieses = [], []
        for i in range(self.n_camadas):
            lim = np.sqrt(6.0 / (camadas[i] + camadas[i + 1]))
            self.mu_pesos.append(g.uniform(-lim, lim, (camadas[i], camadas[i + 1])))
            self.mu_vieses.append(np.zeros(camadas[i + 1]))
            self.log_sigma_pesos.append(np.full((camadas[i], camadas[i + 1]), -2.0))
            self.log_sigma_vieses.append(np.full(camadas[i + 1], -2.0))

        self.rng = np.random.default_rng(semente + 7)

    def amostrar_pesos(self) -> Tuple[list, list]:
        pesos, vieses = [], []
        for i in range(self.n_camadas):
            sp = np.exp(self.log_sigma_pesos[i])
            sb = np.exp(self.log_sigma_vieses[i])
            pesos.append(self.mu_pesos[i] + sp * self.rng.normal(size=self.mu_pesos[i].shape))
            vieses.append(self.mu_vieses[i] + sb * self.rng.normal(size=self.mu_vieses[i].shape))
        return pesos, vieses

    def forward(self, x: np.ndarray, pesos=None, vieses=None) -> np.ndarray:
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if pesos is None:
            pesos, vieses = self.amostrar_pesos()
        a = x
        for i in range(self.n_camadas):
            z = a @ pesos[i] + vieses[i]
            a = np.tanh(z) if i < self.n_camadas - 1 else z
        return a.squeeze()

    def prever_media_var(
        self,
        x: np.ndarray,
        n_amostras: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Média e variância epistêmica sobre amostras do posterior."""
        preds = np.stack([self.forward(x) for _ in range(n_amostras)], axis=0)
        return preds.mean(axis=0), preds.var(axis=0)

    def parametros_vetor(self) -> np.ndarray:
        partes = []
        for i in range(self.n_camadas):
            partes += [
                self.mu_pesos[i].ravel(),
                self.mu_vieses[i].ravel(),
                self.log_sigma_pesos[i].ravel(),
                self.log_sigma_vieses[i].ravel(),
            ]
        return np.concatenate(partes)

    def carregar_parametros(self, theta: np.ndarray) -> None:
        idx = 0
        for i in range(self.n_camadas):
            for attr, shape in [
                ("mu_pesos", self.mu_pesos[i].shape),
                ("mu_vieses", self.mu_vieses[i].shape),
                ("log_sigma_pesos", self.log_sigma_pesos[i].shape),
                ("log_sigma_vieses", self.log_sigma_vieses[i].shape),
            ]:
                n = int(np.prod(shape))
                getattr(self, attr)[i] = theta[idx:idx + n].reshape(shape)
                idx += n

    def n_parametros(self) -> int:
        return len(self.parametros_vetor())

    def kl_prior(self) -> float:
        """KL(q || p) estável, p = N(0, prior_std²)."""
        kl = 0.0
        ps2 = self.prior_std ** 2
        for i in range(self.n_camadas):
            for mu, log_s in [
                (self.mu_pesos[i], self.log_sigma_pesos[i]),
                (self.mu_vieses[i], self.log_sigma_vieses[i]),
            ]:
                log_s = np.clip(log_s, -5.0, -0.5)
                s2 = np.exp(2.0 * log_s)
                kl += 0.5 * np.sum((mu ** 2 + s2) / ps2 - 1.0 - 2.0 * log_s + np.log(ps2))
        return float(np.clip(kl, 0.0, 1e6))
