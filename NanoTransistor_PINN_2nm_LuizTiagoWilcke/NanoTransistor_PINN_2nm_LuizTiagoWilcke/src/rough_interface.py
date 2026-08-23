"""
Módulo: Rugosidade de Interface (analogia a volatilidade rough / fBm)
Autor: Luiz Tiago Wilcke
"""

import torch
import math


class RugosidadeInterface:
    """
    Modelo de rugosidade de interface Si/óxido via movimento browniano fracionário
    (exponente de Hurst H < 0.5 → anti-persistente / rough).
    """
    def __init__(self, H: float = 0.3, amplitude_nm: float = 0.2):
        self.H = H
        self.amp = amplitude_nm

    def gerar_perfil(self, n_pontos: int = 256, seed: int = 42) -> torch.Tensor:
        torch.manual_seed(seed)
        # aproximação simples de fBm via integração fracionária de ruído branco
        ruido = torch.randn(n_pontos)
        # kernel de poder (simplificado)
        k = torch.arange(1, n_pontos+1).float() ** (self.H - 0.5)
        perfil = torch.fft.irfft(torch.fft.rfft(ruido) * k[:len(torch.fft.rfft(ruido))])
        perfil = self.amp * (perfil - perfil.mean()) / (perfil.std() + 1e-8)
        return perfil
