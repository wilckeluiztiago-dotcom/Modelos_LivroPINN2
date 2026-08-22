# -*- coding: utf-8 -*-
"""
Módulo 23: Pipeline de Treinamento Híbrido Adam + L-BFGS
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from torch.optim import Adam, LBFGS
from typing import Callable

def treinar_hibrido(
    modelo: torch.nn.Module,
    perda_fn: Callable,
    n_adam: int = 500,
    n_lbfgs: int = 50,
    lr_adam: float = 1e-3
) -> None:
    """Treinamento em duas fases: Adam exploratório + L-BFGS de alta precisão."""
    opt_adam = Adam(modelo.parameters(), lr=lr_adam)
    for epoch in range(n_adam):
        opt_adam.zero_grad()
        loss = perda_fn()
        loss.backward()
        opt_adam.step()
        if (epoch + 1) % 100 == 0:
            print(f"  [Adam] Época {epoch+1}/{n_adam} | Perda = {loss.item():.6e}")

    opt_lbfgs = LBFGS(
        modelo.parameters(),
        max_iter=20,
        history_size=30,
        line_search_fn="strong_wolfe"
    )

    def closure():
        opt_lbfgs.zero_grad()
        loss = perda_fn()
        loss.backward()
        return loss

    for i in range(n_lbfgs):
        loss = opt_lbfgs.step(closure)
        if (i + 1) % 10 == 0:
            print(f"  [L-BFGS] Iteração {i+1}/{n_lbfgs} | Perda = {loss.item():.6e}")
