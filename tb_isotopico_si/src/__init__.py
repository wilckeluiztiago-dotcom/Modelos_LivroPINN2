"""
TB Isotópico Si²⁸/²⁹/³⁰ · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_iso import (
    amostrar_massas, delta_epsilon, hamiltoniano_iso,
    espectro_ensemble, parametros_iso_default, M_BAR,
)
from .rede_pinn_iso import RedePINN_Iso
from .treinamento_iso import treinar_iso

__all__ = [
    "amostrar_massas",
    "delta_epsilon",
    "hamiltoniano_iso",
    "espectro_ensemble",
    "parametros_iso_default",
    "M_BAR",
    "RedePINN_Iso",
    "treinar_iso",
]
