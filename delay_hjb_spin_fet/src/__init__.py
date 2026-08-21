"""
Delay-HJB com Espaço Estendido — Inércia de Spin em Spin-FETs 2D
Capítulo 37 — Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .dinamica_spin_retardada import DinamicaSpinRetardada
from .hjbd_retardado import residuo_delay_hjb, hamiltoniano_stt
from .rede_pinn_hjb import RedePINN3D
from .treinamento_hjb import treinar_delay_hjb

__all__ = [
    "DinamicaSpinRetardada",
    "residuo_delay_hjb",
    "hamiltoniano_stt",
    "RedePINN3D",
    "treinar_delay_hjb",
]
