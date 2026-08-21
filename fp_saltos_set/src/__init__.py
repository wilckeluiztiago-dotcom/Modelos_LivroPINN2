"""
Fokker–Planck com Saltos Discretos para Memórias SET (Capítulo 8)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .set_carga import simular_set, taxas_tunelamento
from .rede_pinn_set import RedePINN_SET
from .treinamento_set import treinar_set

__all__ = ["simular_set", "taxas_tunelamento", "RedePINN_SET", "treinar_set"]
