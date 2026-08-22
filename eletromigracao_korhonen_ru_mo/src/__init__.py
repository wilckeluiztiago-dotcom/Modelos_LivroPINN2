"""
Eletromigração Korhonen · Ru/Mo · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_em import RedePotencial, RedeTensao
from .treinamento_em import treinar_em
from .fisica_korhonen import parametros_korhonen_default

__all__ = [
    "RedePotencial",
    "RedeTensao",
    "treinar_em",
    "parametros_korhonen_default",
]
