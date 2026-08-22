"""
Cross-Tunneling Leakage · Nanofios Acoplados · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_tunel import RedePINN_Tunel
from .treinamento_tunel import treinar_tunel
from .fisica_tunel import parametros_tunel_default

__all__ = ["RedePINN_Tunel", "treinar_tunel", "parametros_tunel_default"]
