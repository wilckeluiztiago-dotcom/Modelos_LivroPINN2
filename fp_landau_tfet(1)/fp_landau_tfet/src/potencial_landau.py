"""
Potencial quártico de Landau (bimodal) e tempo de escape de Kramers.
Capítulo 41 & Apêndice J.2.
"""

import numpy as np
from typing import Tuple


def potencial_landau(
    x: np.ndarray,
    a: float = 1.0,
    b: float = 1.0,
) -> np.ndarray:
    """
    V(x) = −(a/2) x² + (b/4) x⁴
    Mínimos em ±sqrt(a/b), barreira em x=0 com ΔV = a²/(4b).
    """
    return -0.5 * a * x ** 2 + 0.25 * b * x ** 4


def forca_landau(x: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """F = −V' = a x − b x³."""
    return a * x - b * x ** 3


def derivada_segunda_V(x: np.ndarray, a: float = 1.0, b: float = 1.0) -> np.ndarray:
    """V''(x) = −a + 3 b x²."""
    return -a + 3.0 * b * x ** 2


def barreira_e_minimos(a: float = 1.0, b: float = 1.0) -> Tuple[float, float, float]:
    """
    Retorna (x_min, x_max_barreira, ΔV).
    x_min = sqrt(a/b), barreira em 0, ΔV = a²/(4b).
    """
    x_min = np.sqrt(a / b)
    delta_V = (a ** 2) / (4.0 * b)
    return float(x_min), 0.0, float(delta_V)


def tempo_kramers(
    a: float = 1.0,
    b: float = 1.0,
    sigma: float = 0.4,
) -> float:
    """
    Tempo de escape de Kramers (sobre-amortecido):

        τ_K ≈ 2π / sqrt(V''(x_min) |V''(x_max)|) · exp(2 ΔV / σ²)

    Com V''(x_min) = 2a, V''(0) = −a → sqrt = a√2.
    """
    x_min, _, delta_V = barreira_e_minimos(a, b)
    Vpp_min = float(derivada_segunda_V(np.array([x_min]), a, b)[0])   # 2a
    Vpp_max = float(derivada_segunda_V(np.array([0.0]), a, b)[0])     # −a
    prefactor = 2.0 * np.pi / np.sqrt(Vpp_min * np.abs(Vpp_max))
    return float(prefactor * np.exp(2.0 * delta_V / (sigma ** 2)))
