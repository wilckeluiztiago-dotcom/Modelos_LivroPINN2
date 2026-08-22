# -*- coding: utf-8 -*-
"""
Módulo 01: Constantes Físicas Fundamentais
Autor: Luiz Tiago Wilcke
Descrição: Tensores PyTorch (float64) das constantes SI para modelagem
           de transporte quântico em SET sob a Teoria Ortodoxa.
"""

from __future__ import annotations
import torch
from typing import Final, List

# Precisão global
DTYPE: Final = torch.float64
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constantes fundamentais (SI)
e: Final[torch.Tensor] = torch.tensor(1.602176634e-19, dtype=DTYPE, device=DEVICE)  # Carga elementar [C]
hbar: Final[torch.Tensor] = torch.tensor(1.054571817e-34, dtype=DTYPE, device=DEVICE)  # ħ [J·s]
k_B: Final[torch.Tensor] = torch.tensor(1.380649e-23, dtype=DTYPE, device=DEVICE)  # Boltzmann [J/K]
m_0: Final[torch.Tensor] = torch.tensor(9.1093837015e-31, dtype=DTYPE, device=DEVICE)  # Massa do elétron [kg]
epsilon_0: Final[torch.Tensor] = torch.tensor(8.8541878128e-12, dtype=DTYPE, device=DEVICE)  # Permissividade vácuo [F/m]
epsilon_r_SiO2: Final[torch.Tensor] = torch.tensor(3.9, dtype=DTYPE, device=DEVICE)  # Relativa típica de óxido

# Derivadas úteis
e2: Final[torch.Tensor] = e ** 2
hbar2: Final[torch.Tensor] = hbar ** 2

def para_dispositivo(tensores: List[torch.Tensor]) -> List[torch.Tensor]:
    """Move lista de tensores para o dispositivo ativo."""
    return [t.to(DEVICE) for t in tensores]
