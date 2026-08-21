"""
Geometria 3D de um GAAFET (Gate-All-Around FET).
Canal nanowire cilíndrico envolvido pelo gate.
"""

import numpy as np
from typing import Tuple, Optional


class GeometriaGAAFET:
    """
    Nanoestrutura Gate-All-Around simplificada.

    - Canal: cilindro ao longo de x, raio R_canal
    - Óxido: casca cilíndrica R_canal < r < R_ox
    - Gate: envoltório metálico em r = R_ox (potencial fixo)
    - Source/Drain: faces x=0 e x=L

    Domínio de simulação: caixa [0,L] x [-R,R] x [-R,R]
    com R = R_ox * 1.1
    """

    def __init__(
        self,
        L: float = 1.0,          # comprimento do canal (normalizado ~ nm)
        R_canal: float = 0.25,
        R_ox: float = 0.40,
        V_gate: float = 0.5,
        V_source: float = 0.0,
        V_drain: float = 0.3,
        epsilon_si: float = 11.7,
        epsilon_ox: float = 3.9,
    ):
        self.L = L
        self.R_canal = R_canal
        self.R_ox = R_ox
        self.V_gate = V_gate
        self.V_source = V_source
        self.V_drain = V_drain
        self.epsilon_si = epsilon_si
        self.epsilon_ox = epsilon_ox
        self.R_box = R_ox * 1.15

    def limites_dominio(self) -> np.ndarray:
        """Retorna (3, 2): [xmin,xmax], [ymin,ymax], [zmin,zmax]."""
        return np.array([
            [0.0, self.L],
            [-self.R_box, self.R_box],
            [-self.R_box, self.R_box],
        ])

    def raio_radial(self, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return np.sqrt(y ** 2 + z ** 2)

    def mascara_canal(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        r = self.raio_radial(y, z)
        return (r <= self.R_canal) & (x >= 0) & (x <= self.L)

    def mascara_oxido(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        r = self.raio_radial(y, z)
        return (r > self.R_canal) & (r <= self.R_ox) & (x >= 0) & (x <= self.L)

    def permitividade(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """ε(x,y,z) por região (Si no canal, oxido fora)."""
        eps = np.full_like(x, self.epsilon_ox, dtype=float)
        eps[self.mascara_canal(x, y, z)] = self.epsilon_si
        return eps

    def densidade_carga(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        rho0: float = 1.0,
    ) -> np.ndarray:
        """
        Densidade de carga simplificada no canal
        (dopagem / inversão aproximada).
        """
        rho = np.zeros_like(x)
        mask = self.mascara_canal(x, y, z)
        # perfil suave ao longo do canal
        rho[mask] = rho0 * np.exp(-((x[mask] - self.L / 2) / (self.L / 3)) ** 2)
        return rho

    def pontos_contorno_gate(self, n: int, semente: Optional[int] = 1) -> Tuple[np.ndarray, np.ndarray]:
        """Pontos na superfície do gate (r ≈ R_ox) com φ = V_gate."""
        gerador = np.random.default_rng(semente)
        theta = gerador.uniform(0, 2 * np.pi, n)
        x = gerador.uniform(0, self.L, n)
        y = self.R_ox * np.cos(theta)
        z = self.R_ox * np.sin(theta)
        pts = np.column_stack([x, y, z])
        vals = np.full(n, self.V_gate)
        return pts, vals

    def pontos_contorno_source_drain(self, n: int, semente: Optional[int] = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Faces source (x=0) e drain (x=L)."""
        gerador = np.random.default_rng(semente)
        n2 = n // 2
        # source
        r_s = gerador.uniform(0, self.R_ox, n2)
        th_s = gerador.uniform(0, 2 * np.pi, n2)
        pts_s = np.column_stack([
            np.zeros(n2),
            r_s * np.cos(th_s),
            r_s * np.sin(th_s),
        ])
        vals_s = np.full(n2, self.V_source)
        # drain
        r_d = gerador.uniform(0, self.R_ox, n - n2)
        th_d = gerador.uniform(0, 2 * np.pi, n - n2)
        pts_d = np.column_stack([
            np.full(n - n2, self.L),
            r_d * np.cos(th_d),
            r_d * np.sin(th_d),
        ])
        vals_d = np.full(n - n2, self.V_drain)
        return np.vstack([pts_s, pts_d]), np.concatenate([vals_s, vals_d])
