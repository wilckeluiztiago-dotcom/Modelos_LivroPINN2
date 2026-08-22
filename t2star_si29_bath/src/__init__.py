"""
T₂* Dephasing · ³¹P + banho ²⁹Si · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_banho import (
    gerar_banho_29Si, acoplamentos_dipolares, T2_star_de_A, fid_gaussiano,
)
from .rede_pinn_t2 import RedePINN_T2
from .treinamento_t2 import treinar_t2

__all__ = [
    "gerar_banho_29Si",
    "acoplamentos_dipolares",
    "T2_star_de_A",
    "fid_gaussiano",
    "RedePINN_T2",
    "treinar_t2",
]
