"""
Hidrodinâmico Portadores Quentes · Baccarani–Wordeman · PINN
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_hd import RedePINN_HD
from .treinamento_hd import treinar_hd
from .fisica_hd import parametros_hd_default

__all__ = ["RedePINN_HD", "treinar_hd", "parametros_hd_default"]
