"""
Módulo: Fourier Neural Operator (FNO) para curvas I-V rápidas
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import torch.fft


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.modes = modes
        self.scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: [B, C, N]
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(B, self.weights.shape[1], N//2+1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes] = torch.einsum("bci,ioi->boi", x_ft[:, :, :self.modes], self.weights)
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class FNO1d(nn.Module):
    """Fourier Neural Operator 1D para mapear bias → corrente."""
    def __init__(self, modes=16, width=32, in_dim=1, out_dim=1):
        super().__init__()
        self.fc0 = nn.Linear(in_dim, width)
        self.conv1 = SpectralConv1d(width, width, modes)
        self.conv2 = SpectralConv1d(width, width, modes)
        self.fc1 = nn.Linear(width, 64)
        self.fc2 = nn.Linear(64, out_dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, N, in_dim]
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        x = self.act(self.conv1(x)) + x
        x = self.act(self.conv2(x)) + x
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        return self.fc2(x)
