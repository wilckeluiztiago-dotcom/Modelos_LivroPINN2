"""
Módulo: Flutuação de Dopantes Aleatórios (RDF)
Autor: Luiz Tiago Wilcke
"""

import torch
import math


class RuidoRDF:
    """
    Modelo simplificado de Random Dopant Fluctuation.
    Adiciona ruído espacial gaussiano à dopagem com amplitude ~ 1/sqrt(N·Vol).
    """
    def __init__(self, volume_m3: float, seed: int = 42):
        self.volume = volume_m3
        self.gerador = torch.Generator().manual_seed(seed)

    def aplicar(self, Nd: torch.Tensor, amplitude: float = 0.1) -> torch.Tensor:
        """Nd_ruidoso = Nd * (1 + amplitude * ε), ε ~ N(0,1)"""
        ruido = torch.randn(Nd.shape, generator=self.gerador, device=Nd.device)
        # amplitude física ~ 1/sqrt(N·V)
        sigma_fisico = amplitude / torch.sqrt(torch.clamp(Nd * self.volume, min=1.0))
        return Nd * (1.0 + sigma_fisico * ruido)

    def variancia_relativa(self, Nd: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(torch.clamp(Nd * self.volume, min=1.0))
