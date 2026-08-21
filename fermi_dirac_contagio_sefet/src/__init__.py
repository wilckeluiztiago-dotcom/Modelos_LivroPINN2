"""
Contágio Fermi–Dirac em Cadeias de Dopantes — Single-Electron FETs
Apêndice J.4 — Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .cadeia_dopantes import CadeiaDopantes
from .contagio_fermi_dirac import (
    probabilidade_fermi_dirac,
    passo_contagio,
    simular_transporte,
)
from .pinn_condutancia import RedePINN, treinar_condutancia

__all__ = [
    "CadeiaDopantes",
    "probabilidade_fermi_dirac",
    "passo_contagio",
    "simular_transporte",
    "RedePINN",
    "treinar_condutancia",
]
