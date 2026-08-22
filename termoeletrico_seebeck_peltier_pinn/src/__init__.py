"""
Transporte Termoelétrico Não-Linear (Seebeck / Peltier) · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_termo import RedePINN_Termo
from .treinamento_termo import treinar_termo
from .fisica_termo import parametros_termo_default

__all__ = [
    "RedePINN_Termo",
    "treinar_termo",
    "parametros_termo_default",
]
