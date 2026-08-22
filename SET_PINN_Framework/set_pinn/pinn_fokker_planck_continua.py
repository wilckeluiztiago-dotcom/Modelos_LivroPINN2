# -*- coding: utf-8 -*-
"""
Módulo 11: PINN para Fokker-Planck Contínua de Carga
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .redes_base_ativacoes import criar_mlp
from .operador_autograd_quantico import gradientes_alta_ordem
from .constantes_fisicas import DTYPE, DEVICE

class PINNFokkerPlanck(nn.Module):
    def __init__(self, camadas: list = None):
        super().__init__()
        if camadas is None:
            camadas = [128, 128, 128]
        self.rede = criar_mlp(4, 1, camadas, ativacao="swish")

    def forward(self, entradas: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.rede(entradas))  # densidade positiva

    def residuo_fp(
        self,
        entradas: torch.Tensor,
        D1: torch.Tensor,
        D2: torch.Tensor
    ) -> torch.Tensor:
        p = self.forward(entradas)
        dt, dq, dqq, _, _ = gradientes_alta_ordem(p, entradas)
        # ∂p/∂t + ∂(D1 p)/∂q - ½ ∂²(D2 p)/∂q² ≈ 0
        fluxo = D1 * p
        d_fluxo = torch.autograd.grad(
            fluxo.sum(), entradas, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        difusao = 0.5 * D2 * p
        d_dif = torch.autograd.grad(
            difusao.sum(), entradas, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        d2_dif = torch.autograd.grad(
            d_dif.sum(), entradas, create_graph=True, retain_graph=True
        )[0][:, 1:2]
        return dt + d_fluxo - d2_dif
