# -*- coding: utf-8 -*-
"""
Módulo 24: Extração Analítica de Gregas Quânticas via Autograd
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Tuple

def calcular_gregas(
    I: torch.Tensor,
    V_D: torch.Tensor,
    V_G: torch.Tensor,
    T: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    g_m = ∂I/∂V_G ,  g_ds = ∂I/∂V_D ,  sensibilidade térmica ∂I/∂T
    """
    gm = torch.autograd.grad(I, V_G, grad_outputs=torch.ones_like(I), create_graph=True, allow_unused=True)[0]
    gds = torch.autograd.grad(I, V_D, grad_outputs=torch.ones_like(I), create_graph=True, allow_unused=True)[0]
    dIdT = torch.autograd.grad(I, T, grad_outputs=torch.ones_like(I), create_graph=True, allow_unused=True)[0]
    if gm is None:
        gm = torch.zeros_like(I)
    if gds is None:
        gds = torch.zeros_like(I)
    if dIdT is None:
        dIdT = torch.zeros_like(I)
    return gm, gds, dIdT
