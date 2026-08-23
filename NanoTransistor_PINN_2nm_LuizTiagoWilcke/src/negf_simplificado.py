"""
Módulo: NEGF Simplificado (densidade espectral aproximada)
Autor: Luiz Tiago Wilcke
"""

import torch
import math


class NEGFSimplificado:
    """
    Aproximação de densidade espectral e função de Green para canal curto.
    Não é um solver NEGF completo, mas fornece termos de autoenergia para PINN.
    """
    def __init__(self, gamma_S: float = 0.1, gamma_D: float = 0.1):
        self.gamma_S = gamma_S  # eV
        self.gamma_D = gamma_D

    def densidade_espectral(self, E: torch.Tensor, E0: float = 0.0):
        """A(E) ≈ (γ/π) / ((E-E0)² + γ²)  (Lorentziana)"""
        gamma = 0.5 * (self.gamma_S + self.gamma_D)
        A = (gamma / math.pi) / ((E - E0)**2 + gamma**2)
        return A

    def corrente_aproximada(self, n_S, n_D, Vds, T=300.0):
        """Corrente aproximada por diferença de ocupação."""
        return (n_S - n_D) * Vds * 1e-4  # escala empírica
