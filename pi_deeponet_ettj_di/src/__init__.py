"""
PI-DeepONet para ETTJ DI Futuro B3
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .curva_di_b3 import gerar_curva_di, gerar_superficie_P, VERTICES_DI
from .rede_deeponet import PIDeepONet
from .treinamento_deeponet import treinar_deeponet

__all__ = [
    "gerar_curva_di",
    "gerar_superficie_P",
    "VERTICES_DI",
    "PIDeepONet",
    "treinar_deeponet",
]
