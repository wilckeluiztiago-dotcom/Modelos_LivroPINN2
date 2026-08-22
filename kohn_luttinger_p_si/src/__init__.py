"""
Kohn–Luttinger / EMA + célula central · ³¹P:Si · PINN
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .rede_pinn_ema import BancoEMA
from .treinamento_ema import treinar_ema
from .fisica_ema import parametros_ema_default, E_BIND

__all__ = ["BancoEMA", "treinar_ema", "parametros_ema_default", "E_BIND"]
