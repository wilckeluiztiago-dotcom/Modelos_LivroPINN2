"""
PIDE · Derivativos de Energia / Opções sobre PLD
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .pld_hidrologia import simular_pld, theta_sazonal
from .rede_pinn_pide import RedePINN_PIDE
from .treinamento_pide import treinar_pide

__all__ = ["simular_pld", "theta_sazonal", "RedePINN_PIDE", "treinar_pide"]
