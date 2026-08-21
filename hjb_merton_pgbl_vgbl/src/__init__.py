"""
HJB-Merton · Portfólios PGBL/VGBL
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .merton_crra import pi_otimo_merton, c_otimo_aprox, simular_riqueza
from .rede_pinn_hjb import RedePINN_HJB
from .treinamento_hjb import treinar_hjb

__all__ = [
    "pi_otimo_merton",
    "c_otimo_aprox",
    "simular_riqueza",
    "RedePINN_HJB",
    "treinar_hjb",
]
