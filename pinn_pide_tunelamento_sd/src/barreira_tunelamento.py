"""
Barreira source–drain e kernel de tunelamento quântico (sub-12 nm).
"""

import numpy as np
from typing import Tuple, Optional


class CanalSub12nm:
    """
    Canal de transistor com comprimento L < 12 nm (unidades normalizadas:
    L=1 corresponde a ~10 nm físico). Tunelamento source–drain relevante.
    """

    def __init__(
        self,
        L: float = 1.0,           # ~10 nm
        V_barreira: float = 0.35,
        x_centro: float = 0.5,
        largura: float = 0.18,
        V_source: float = 0.0,
        V_drain: float = 0.25,
        m_eff: float = 1.0,
        hbar: float = 1.0,
    ):
        self.L = L
        self.V_barreira = V_barreira
        self.x_centro = x_centro
        self.largura = largura
        self.V_source = V_source
        self.V_drain = V_drain
        self.m_eff = m_eff
        self.hbar = hbar

    def potencial(self, x: np.ndarray) -> np.ndarray:
        """Barreira tipo sech² + polarização source–drain."""
        xi = (x - self.x_centro) / self.largura
        Vb = self.V_barreira * (1.0 / np.cosh(xi)) ** 2
        # rampa linear source → drain
        Vr = self.V_source + (self.V_drain - self.V_source) * (x / self.L)
        return Vb + Vr

    def kappa_wkb(self, x: np.ndarray, E: float) -> np.ndarray:
        """
        Número de onda de decaimento WKB:
            κ(x) = sqrt(2m (V(x) - E)) / ℏ    (região classicamente proibida)
        """
        V = self.potencial(x)
        arg = np.maximum(2.0 * self.m_eff * (V - E), 0.0)
        return np.sqrt(arg) / self.hbar

    def transmissao_wkb(self, E: float, n_quad: int = 80) -> float:
        """
        Coeficiente de transmissão WKB aproximado:
            T(E) ≈ exp( -2 ∫_{x1}^{x2} κ(x) dx )
        """
        x = np.linspace(0, self.L, n_quad)
        kappa = self.kappa_wkb(x, E)
        # só integra onde V > E
        dx = x[1] - x[0]
        integral = np.sum(kappa) * dx
        return float(np.exp(-2.0 * integral))


def kernel_tunelamento(
    x: np.ndarray,
    y: np.ndarray,
    canal: CanalSub12nm,
    E_ref: float = 0.1,
    alpha: float = 3.0,
) -> np.ndarray:
    """
    Kernel integro-diferencial simplificado de penetração de barreira.

    K(x,y) ∝ exp( -α ∫_min(x,y)^{max(x,y)} κ(s) ds )

    Representa a amplitude de salto quântico não-local source↔drain
    através da barreira (modelo reduzido de matriz de tunelamento).
    """
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    # grade auxiliar
    n = 40
    s = np.linspace(0, canal.L, n)
    kappa = canal.kappa_wkb(s, E_ref)
    # integral cumulativa de κ
    dx = s[1] - s[0]
    Kcum = np.concatenate([[0.0], np.cumsum(kappa[:-1]) * dx])

    def integ_abs(a, b):
        # |∫_a^b κ|
        ia = np.interp(a, s, Kcum)
        ib = np.interp(b, s, Kcum)
        return np.abs(ib - ia)

    if x.ndim == 0:
        x = np.array([x])
    # broadcasting para pares (x_i, y_j) ou vetores alinhados
    if y.size == 1:
        y = np.full_like(x, float(y))
    integ = np.array([integ_abs(xi, yi) for xi, yi in zip(x, y)])
    return np.exp(-alpha * integ)
