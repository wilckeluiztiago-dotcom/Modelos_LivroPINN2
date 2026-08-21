"""
Modelo LSV + Condição de Gyöngy para Mobilidade de Portadores (Cap. 21)

Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .cir_variancia import simular_cir, passo_cir
from .fator_local import fator_local_L, mobilidade_efetiva_dupire
from .lsv_dinamica import simular_lsv
from .gyongy_calibracao import calibrar_L_gyongy
from .rede_pinn_gyongy import RedePINN1D
from .treinamento_gyongy import treinar_gyongy

__all__ = [
    "simular_cir",
    "passo_cir",
    "fator_local_L",
    "mobilidade_efetiva_dupire",
    "simular_lsv",
    "calibrar_L_gyongy",
    "RedePINN1D",
    "treinar_gyongy",
]
