"""
Módulo 16: Extração de Gregas via Autograd
Autor: Luiz Tiago Wilcke

Gregas do título zero-cupom:
  Duration ≈ - (1/P) ∂P/∂y  (onde y é yield)
  Convexidade, DV01, etc.
"""

import torch
from .arquitetura_deeponet import PIDeepONetHJM
from .config import CONFIG


def calcular_duration(
    modelo: PIDeepONetHJM,
    u: torch.Tensor,
    t: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """
    Duration modificada aproximada:
        D = - (1/P) * ∂P/∂y   com y = -log(P)/(T-t)
    """
    t = t.requires_grad_(True)
    T = T.requires_grad_(True)
    P = modelo(u, t, T)
    dP_dT = torch.autograd.grad(P, T, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    duration = - (T - t) * (dP_dT / (P + 1e-8))
    return duration


def calcular_convexidade(
    modelo: PIDeepONetHJM,
    u: torch.Tensor,
    t: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """Segunda derivada em relação à maturidade."""
    t = t.requires_grad_(True)
    T = T.requires_grad_(True)
    P = modelo(u, t, T)
    dP_dT = torch.autograd.grad(P, T, grad_outputs=torch.ones_like(P), create_graph=True)[0]
    d2P_dT2 = torch.autograd.grad(dP_dT, T, grad_outputs=torch.ones_like(dP_dT), create_graph=True)[0]
    return d2P_dT2 / (P + 1e-8)
