"""
PINN 3D livre de malhas para Equação de Poisson em GAAFETs

Autor: Luiz Tiago Wilcke
Base: Redes Neurais Informadas pela Física — Aplicações no Mercado Financeiro
      (Volume II) e formalismo geral de PINNs (Caps. 2–3)
"""

__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn3d import RedePINN3D
from .geometria_gaafet import GeometriaGAAFET
from .residuo_poisson import perda_composta, residuo_poisson
from .treinamento import treinar_pinn3d
from .utils import amostragem_lhs, erro_l2

__all__ = [
    "RedePINN3D",
    "GeometriaGAAFET",
    "perda_composta",
    "residuo_poisson",
    "treinar_pinn3d",
    "amostragem_lhs",
    "erro_l2",
]
