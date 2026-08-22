"""
Landauer–Büttiker · Condutância Quantizada · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .fisica_landauer import condutancia_vs_gate, corrente_landauer, modos_analiticos_poco
from .rede_pinn_modos import BancoModos
from .treinamento_modos import treinar_modos

__all__ = [
    "condutancia_vs_gate",
    "corrente_landauer",
    "modos_analiticos_poco",
    "BancoModos",
    "treinar_modos",
]
