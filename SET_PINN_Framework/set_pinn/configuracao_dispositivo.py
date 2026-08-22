# -*- coding: utf-8 -*-
"""
Módulo 02: Configuração Geométrica e Parâmetros do SET
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import Dict
from .constantes_fisicas import DTYPE, DEVICE, e

@dataclass
class ConfiguracaoSET:
    """
    Parâmetros eletrostáticos e de tunelamento do Transistor de Elétron Único.
    C_S, C_D, C_G, C_P : capacitâncias [F]
    R_T_S, R_T_D       : resistências de tunelamento [Ω]
    T_e                : temperatura eletrônica [K]
    """
    C_S: torch.Tensor
    C_D: torch.Tensor
    C_G: torch.Tensor
    C_P: torch.Tensor = field(default_factory=lambda: torch.tensor(0.0, dtype=DTYPE, device=DEVICE))
    R_T_S: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0e5, dtype=DTYPE, device=DEVICE))
    R_T_D: torch.Tensor = field(default_factory=lambda: torch.tensor(1.0e5, dtype=DTYPE, device=DEVICE))
    T_e: torch.Tensor = field(default_factory=lambda: torch.tensor(0.1, dtype=DTYPE, device=DEVICE))

    def __post_init__(self) -> None:
        self.C_Sigma = self.C_S + self.C_D + self.C_G + self.C_P
        self.E_C = e**2 / (2.0 * self.C_Sigma)  # Energia de carregamento [J]

    def para_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "C_S": self.C_S, "C_D": self.C_D, "C_G": self.C_G, "C_P": self.C_P,
            "R_T_S": self.R_T_S, "R_T_D": self.R_T_D, "T_e": self.T_e,
            "C_Sigma": self.C_Sigma, "E_C": self.E_C
        }

def criar_configuracao_padrao() -> ConfiguracaoSET:
    """Configuração típica de SET de ilha metálica em temperatura criogênica."""
    return ConfiguracaoSET(
        C_S=torch.tensor(1.0e-16, dtype=DTYPE, device=DEVICE),
        C_D=torch.tensor(1.0e-16, dtype=DTYPE, device=DEVICE),
        C_G=torch.tensor(5.0e-17, dtype=DTYPE, device=DEVICE),
        C_P=torch.tensor(1.0e-17, dtype=DTYPE, device=DEVICE),
        R_T_S=torch.tensor(2.0e5, dtype=DTYPE, device=DEVICE),
        R_T_D=torch.tensor(2.0e5, dtype=DTYPE, device=DEVICE),
        T_e=torch.tensor(0.05, dtype=DTYPE, device=DEVICE)
    )
