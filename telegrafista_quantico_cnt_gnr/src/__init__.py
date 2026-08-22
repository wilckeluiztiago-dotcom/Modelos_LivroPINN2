"""
Telegrafista Quântico · CNT / GNR · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_qtl import RedePINN_QTL
from .treinamento_qtl import treinar_qtl
from .fisica_telegrafista import parametros_qtl_default, impedancia_caracteristica

__all__ = [
    "RedePINN_QTL",
    "treinar_qtl",
    "parametros_qtl_default",
    "impedancia_caracteristica",
]
