"""
DGM 6D para Equação de Wigner–Boltzmann em Nanofolhas de 1.6 nm

Autor: Luiz Tiago Wilcke
"""

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .celula_dgm import RedeDGM, CelulaDGM
from .wigner_boltzmann import NanofolhaWigner
from .residuo_wigner import residuo_wigner_reduzido, perda_composta_wigner
from .treinamento import treinar_dgm
from .utils import amostragem_lhs

__all__ = [
    "RedeDGM",
    "CelulaDGM",
    "NanofolhaWigner",
    "residuo_wigner_reduzido",
    "perda_composta_wigner",
    "treinar_dgm",
    "amostragem_lhs",
]
