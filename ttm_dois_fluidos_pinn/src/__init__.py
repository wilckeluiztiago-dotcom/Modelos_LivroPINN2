"""
TTM · Dois Fluidos Elétron–Fônon · PINN
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_ttm import simular_ttm, parametros_ttm_default
from .rede_pinn_ttm import RedePINN_TTM
from .treinamento_ttm import treinar_ttm

__all__ = [
    "simular_ttm",
    "parametros_ttm_default",
    "RedePINN_TTM",
    "treinar_ttm",
]
