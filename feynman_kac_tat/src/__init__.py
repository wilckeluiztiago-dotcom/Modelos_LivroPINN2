"""
Feynman–Kac com Saltos para TAT (Cap. 17 & Apêndice A.7)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .processo_saltos import simular_trajetorias, passo_jump_diffusion
from .tat_dieletrico import DieletricoTAT
from .rede_pinn_fk import RedePINN_FK
from .treinamento_fk import treinar_fk

__all__ = [
    "simular_trajetorias",
    "passo_jump_diffusion",
    "DieletricoTAT",
    "RedePINN_FK",
    "treinar_fk",
]
