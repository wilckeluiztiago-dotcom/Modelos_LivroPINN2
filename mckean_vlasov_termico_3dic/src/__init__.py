"""
Contágio Térmico McKean–Vlasov em 3D-IC (Cap. 24 & 40)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .mckean_vlasov import simular_populacao, passo_mckean_vlasov
from .rede_pinn_mv import RedePINN_MV
from .treinamento_mv import treinar_mv

__all__ = [
    "simular_populacao",
    "passo_mckean_vlasov",
    "RedePINN_MV",
    "treinar_mv",
]
