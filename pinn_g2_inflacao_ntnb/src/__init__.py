"""
PINN G2++ · Inflação Implícita NTN-B vs DI
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .modelo_g2_inflacao import simular_fatores, inflacao_implicita_breakeven
from .rede_pinn_g2 import RedePINN_G2
from .treinamento_g2 import treinar_g2

__all__ = [
    "simular_fatores",
    "inflacao_implicita_breakeven",
    "RedePINN_G2",
    "treinar_g2",
]
