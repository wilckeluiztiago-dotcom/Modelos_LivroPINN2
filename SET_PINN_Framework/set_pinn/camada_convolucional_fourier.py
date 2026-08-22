# -*- coding: utf-8 -*-
"""
Módulo 14: SpectralConv (Fourier Neural Operator 1D/2D)
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
import torch.nn as nn
from .constantes_fisicas import DTYPE, DEVICE

class SpectralConv1d(nn.Module):
    def __init__(self, canais_in: int, canais_out: int, modos: int):
        super().__init__()
        self.modos = modos
        scale = 1.0 / (canais_in * canais_out)
        self.pesos = nn.Parameter(
            scale * torch.randn(canais_in, canais_out, modos, dtype=torch.cfloat, device=DEVICE)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, canais, comprimento)
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros_like(x_ft)
        out_ft[:, :, :self.modos] = torch.einsum(
            "bix,iox->box", x_ft[:, :, :self.modos], self.pesos
        )
        return torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
