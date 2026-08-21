"""
Perdas do I-GAN: adversária + física (Poisson).
"""

import numpy as np
from typing import Tuple
from .rede_gan import Gerador, Discriminador
from .graos_wff import potencial_de_wf, residuo_poisson_1d


def perda_discriminador(
    D: Discriminador,
    wf_real: np.ndarray,
    wf_fake: np.ndarray,
) -> float:
    """−E[log D(real)] − E[log(1 − D(fake))]."""
    d_real = np.clip(D.discriminar(wf_real), 1e-6, 1 - 1e-6)
    d_fake = np.clip(D.discriminar(wf_fake), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.log(d_real)) - np.mean(np.log(1.0 - d_fake)))


def perda_gerador_adv(
    D: Discriminador,
    wf_fake: np.ndarray,
) -> float:
    """−E[log D(fake)] (não saturante)."""
    d_fake = np.clip(D.discriminar(wf_fake), 1e-6, 1 - 1e-6)
    return float(-np.mean(np.log(d_fake)))


def perda_fisica_poisson(
    wf_fake: np.ndarray,
    epsilon: float = 1.0,
) -> float:
    """
    λ_phys ‖∇·(ε∇φ_G) + ρ‖²
    φ_G derivado do mapa WF gerado; ρ ≈ 0 no canal (vácuo efetivo)
    ou residual da curvatura de φ.
    """
    total = 0.0
    n = wf_fake.shape[0] if wf_fake.ndim == 3 else 1
    maps = wf_fake if wf_fake.ndim == 3 else wf_fake[np.newaxis]
    for i in range(n):
        phi = potencial_de_wf(maps[i])
        rho = np.zeros_like(phi)  # região de depleção efetiva
        total += residuo_poisson_1d(phi, rho, epsilon)
    return total / n


def perda_gerador_total(
    G: Gerador,
    D: Discriminador,
    z: np.ndarray,
    lambda_phys: float = 0.5,
) -> Tuple[float, float, float]:
    wf_fake = G.gerar(z)
    adv = perda_gerador_adv(D, wf_fake)
    phys = perda_fisica_poisson(wf_fake)
    return adv + lambda_phys * phys, adv, phys
