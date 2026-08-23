"""
Módulo: Extração de Gregas (sensibilidades) via Autograd
Autor: Luiz Tiago Wilcke
"""

import torch


def calcular_delta(modelo, x, Vgs_idx=None):
    """Δ = ∂φ/∂Vgs ou ∂n/∂Vgs aproximado."""
    x = x.clone().requires_grad_(True)
    saida = modelo(x)
    phi = saida[:, 0:1]
    # gradiente espacial como proxy de sensibilidade
    dphi_dx = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                                  create_graph=False)[0]
    return dphi_dx


def calcular_gamma(modelo, x):
    """Γ ≈ ∂²φ/∂x² (curvatura)."""
    x = x.clone().requires_grad_(True)
    saida = modelo(x)
    phi = saida[:, 0:1]
    dphi = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi),
                               create_graph=True)[0]
    d2phi = torch.autograd.grad(dphi, x, grad_outputs=torch.ones_like(dphi),
                                create_graph=False)[0]
    return d2phi


def extrair_todas_gregas(modelo, x):
    return {
        "delta": calcular_delta(modelo, x),
        "gamma": calcular_gamma(modelo, x),
    }
