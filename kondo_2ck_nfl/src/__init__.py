"""
2CK Não-Fermi Líquido · Ilhas Quânticas Si · PINN
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_2ck import G_2CK, G_2CK_T, parametros_2ck_default, entropia_residual
from .rede_pinn_2ck import RedePINN_G, RedePINN_Rho
from .treinamento_2ck import treinar_G, treinar_rho

__all__ = [
    "G_2CK", "G_2CK_T", "parametros_2ck_default", "entropia_residual",
    "RedePINN_G", "RedePINN_Rho", "treinar_G", "treinar_rho",
]
