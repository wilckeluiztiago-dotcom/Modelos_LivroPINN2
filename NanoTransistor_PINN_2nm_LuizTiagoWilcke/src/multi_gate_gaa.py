"""
Módulo: Extensão Multi-Gate / GAA Nanosheet
Autor: Luiz Tiago Wilcke
"""

import torch
from geometria_dispositivo import GeometriaNanotransistor


class GeometriaGAA(GeometriaNanotransistor):
    """Especialização para Gate-All-Around com múltiplas folhas."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tipo = "GAA_nanosheet"

    def fator_forma(self) -> float:
        """Fator de controle de porta (aproximado)."""
        return 1.0 + 0.3 * self.numero_folhas

    def capacitancia_oxido(self, epsilon_ox=3.9*8.85e-12) -> float:
        """Cox por unidade de área."""
        t_ox = self.espessura_oxido_nm * 1e-9
        return epsilon_ox / t_ox
