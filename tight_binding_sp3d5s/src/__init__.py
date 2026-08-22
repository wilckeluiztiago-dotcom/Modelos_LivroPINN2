"""
Tight-Binding sp³d⁵s* · P:Si · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_tb import (
    gerar_cluster_diamante, montar_hamiltoniano, diagonalizar_tb,
    ORBITAIS, parametros_tb_default,
)
from .rede_pinn_tb import RedePINN_TB
from .treinamento_tb import treinar_tb

__all__ = [
    "gerar_cluster_diamante",
    "montar_hamiltoniano",
    "diagonalizar_tb",
    "ORBITAIS",
    "parametros_tb_default",
    "RedePINN_TB",
    "treinar_tb",
]
