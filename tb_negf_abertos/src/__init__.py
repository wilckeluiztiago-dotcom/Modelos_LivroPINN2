"""
TB-NEGF Atomístico · Eletrodos Abertos · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_negf import (
    hamiltoniano_canal, green_retardada, transmissao, corrente_landauer,
    parametros_negf_default,
)
from .rede_pinn_negf import RedePINN_NEGF
from .treinamento_negf import treinar_negf

__all__ = [
    "hamiltoniano_canal",
    "green_retardada",
    "transmissao",
    "corrente_landauer",
    "parametros_negf_default",
    "RedePINN_NEGF",
    "treinar_negf",
]
