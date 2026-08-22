"""
DBRTD · Tsu–Esaki · Dupla Barreira · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_dbrtd import (
    potencial_dupla_barreira, transmissao_transfer_matrix,
    corrente_tsu_esaki, curva_JV, parametros_dbrtd_default,
)
from .rede_pinn_dbrtd import RedePINN_DBRTD
from .treinamento_dbrtd import treinar_dbrtd

__all__ = [
    "potencial_dupla_barreira",
    "transmissao_transfer_matrix",
    "corrente_tsu_esaki",
    "curva_JV",
    "parametros_dbrtd_default",
    "RedePINN_DBRTD",
    "treinar_dbrtd",
]
