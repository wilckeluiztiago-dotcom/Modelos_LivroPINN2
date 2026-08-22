# -*- coding: utf-8 -*-
"""
Módulo 09: Extração de Derivadas de Alta Ordem via Autograd
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from typing import Tuple

def gradientes_alta_ordem(
    u: torch.Tensor,
    entradas: torch.Tensor,
    criar_grafo: bool = True
) -> Tuple[torch.Tensor, ...]:
    """
    Retorna ∂u/∂t, ∂u/∂q, ∂²u/∂q², ∂u/∂V_D, ∂u/∂V_G
    (assumindo entradas = [t, q, V_D, V_G]).
    """
    grad_u = torch.autograd.grad(
        u, entradas, grad_outputs=torch.ones_like(u),
        create_graph=criar_grafo, retain_graph=True
    )[0]
    du_dt = grad_u[:, 0:1]
    du_dq = grad_u[:, 1:2]
    du_dVD = grad_u[:, 2:3]
    du_dVG = grad_u[:, 3:4]

    d2u_dq2 = torch.autograd.grad(
        du_dq, entradas, grad_outputs=torch.ones_like(du_dq),
        create_graph=criar_grafo, retain_graph=True
    )[0][:, 1:2]

    return du_dt, du_dq, d2u_dq2, du_dVD, du_dVG
