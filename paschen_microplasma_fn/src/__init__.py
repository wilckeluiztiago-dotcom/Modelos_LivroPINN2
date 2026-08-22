"""
Paschen Modificado · Microplasma FN · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_fn import RedePINN_FN
from .treinamento_fn import treinar_fn
from .fisica_paschen import parametros_fn_default, tensao_paschen_classica, tensao_fn_gap

__all__ = [
    "RedePINN_FN",
    "treinar_fn",
    "parametros_fn_default",
    "tensao_paschen_classica",
    "tensao_fn_gap",
]
