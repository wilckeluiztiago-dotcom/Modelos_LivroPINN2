# -*- coding: utf-8 -*-
"""
Módulo 04: Distribuição de Fermi-Dirac com Regularização Suave
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .constantes_fisicas import k_B, DTYPE

def fermi_dirac(energia: torch.Tensor, mu: torch.Tensor, T: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    f(ε) = 1 / (1 + exp((ε - μ)/(k_B T)))
    Regularização contra overflow quando T → 0.
    """
    x = (energia - mu) / (k_B * T + eps)
    x = torch.clamp(x, -50.0, 50.0)
    return 1.0 / (1.0 + torch.exp(x))

def integrando_tunelamento(delta_F: torch.Tensor, T: torch.Tensor, eps: float = 1e-14) -> torch.Tensor:
    """
    Forma fechada da integral Fermi-Dirac:
    ∫ f(ε)[1-f(ε+ΔF)] dε = -ΔF / (1 - exp(ΔF / k_B T))
    """
    x = delta_F / (k_B * T + eps)
    x = torch.clamp(x, -80.0, 80.0)
    return -delta_F / (1.0 - torch.exp(x) + eps)
