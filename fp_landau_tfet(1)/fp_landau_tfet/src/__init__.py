"""
Fokker–Planck + Landau para chaveamento TFET (Cap. 41 & Apêndice J.2)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .potencial_landau import potencial_landau, tempo_kramers, barreira_e_minimos
from .langevin_fp import simular_langevin, densidade_estacionaria_analitica
from .rede_pinn_fp import RedePINN_FP
from .treinamento_fp import treinar_fp

__all__ = [
    "potencial_landau",
    "tempo_kramers",
    "barreira_e_minimos",
    "simular_langevin",
    "densidade_estacionaria_analitica",
    "RedePINN_FP",
    "treinar_fp",
]
