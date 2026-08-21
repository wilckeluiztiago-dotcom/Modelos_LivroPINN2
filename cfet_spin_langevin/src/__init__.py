"""
Acoplamento Espin–Langevin para CFETs Quânticos (Apêndice J.3)

Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .ising_glauber import RedeIsingGlauber
from .langevin_potencial import ProcessoLangevin, potencial_nao_linear
from .acoplamento_cfet import CFETSpinLangevin

__all__ = [
    "RedeIsingGlauber",
    "ProcessoLangevin",
    "potencial_nao_linear",
    "CFETSpinLangevin",
]
