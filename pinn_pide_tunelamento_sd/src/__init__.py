"""
PINN Integro-Diferencial (PIDE) para Tunelamento Source–Drain sub-12 nm

Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn import RedePINN
from .barreira_tunelamento import CanalSub12nm, kernel_tunelamento
from .residuo_pide import residuo_pide_estacionario, perda_pide
from .treinamento import treinar_pide

__all__ = [
    "RedePINN",
    "CanalSub12nm",
    "kernel_tunelamento",
    "residuo_pide_estacionario",
    "perda_pide",
    "treinar_pide",
]
