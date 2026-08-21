"""
I-GANs para Síntese de WFF Metálico (Capítulo 44)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .graos_wff import gerar_mapa_graos, potencial_de_wf
from .rede_gan import Gerador, Discriminador
from .treinamento_igan import treinar_igan

__all__ = [
    "gerar_mapa_graos",
    "potencial_de_wf",
    "Gerador",
    "Discriminador",
    "treinar_igan",
]
