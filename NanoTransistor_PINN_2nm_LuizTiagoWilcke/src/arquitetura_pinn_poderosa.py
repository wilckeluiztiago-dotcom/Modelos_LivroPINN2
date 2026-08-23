"""
Módulo: Arquitetura PINN Poderosa – Residual + Fourier Features + Softplus
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 1, num_freq: int = 64, scale: float = 12.0):
        super().__init__()
        B = torch.randn(in_dim, num_freq) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.Tanh()
        # residual scaling para estabilidade
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(residual + self.alpha * out)


class PINNPoderosa(nn.Module):
    """
    Rede multi-saída:
      saída[:,0] → φ* (potencial normalizado)
      saída[:,1] → n* (densidade normalizada, depois softplus)
    """
    def __init__(self, in_dim: int = 1, out_dim: int = 2, hidden: int = 192,
                 n_blocks: int = 5, fourier: bool = True, num_freq: int = 48):
        super().__init__()
        self.fourier = fourier
        if fourier:
            self.ff = FourierFeatures(in_dim, num_freq=num_freq)
            first_dim = 2 * num_freq
        else:
            first_dim = in_dim

        layers = [nn.Linear(first_dim, hidden), nn.Tanh()]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
        self._inicializar()

    def _inicializar(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.ff(coords) if self.fourier else coords
        return self.net(x)

    def num_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
