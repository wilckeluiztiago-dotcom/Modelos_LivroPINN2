"""
PI-DRL Market Making WIN/WDO (Avellaneda–Stoikov)
Autor: Luiz Tiago Wilcke
"""
__version__ = "1.0.0"
__author__ = "Luiz Tiago Wilcke"

from .avellaneda_stoikov import simular_mm, reservas_as
from .rede_pidrl import Critic, Actor
from .treinamento_pidrl import treinar_pidrl

__all__ = ["simular_mm", "reservas_as", "Critic", "Actor", "treinar_pidrl"]
