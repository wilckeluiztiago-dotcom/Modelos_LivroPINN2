"""
Módulo: Arquitetura PINN Poderosa (Residual MLP + Fourier Features)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import math


class FourierFeatures(nn.Module):
    def __init__(self, in_dim: int = 1, num_freq: int = 64, scale: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_freq) * scale
        self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, ativacao=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = ativacao()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)


class PINNPoderosa(nn.Module):
    """
    Rede multi-saída para (φ, n, ψ, ...) 
    Entrada: coordenadas normalizadas (x, y, Vgs, Vds, ...)
    """
    def __init__(self, in_dim: int = 2, out_dim: int = 3, hidden: int = 256,
                 n_blocks: int = 6, fourier: bool = True, num_freq: int = 64):
        super().__init__()
        self.fourier = fourier
        if fourier:
            self.ff = FourierFeatures(in_dim, num_freq=num_freq)
            first_dim = 2 * num_freq
        else:
            first_dim = in_dim

        camadas = [nn.Linear(first_dim, hidden), nn.Tanh()]
        for _ in range(n_blocks):
            camadas.append(ResidualBlock(hidden))
        camadas.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*camadas)

        self._inicializar()

    def _inicializar(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        if self.fourier:
            x = self.ff(coords)
        else:
            x = coords
        return self.net(x)

    def num_parametros(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    modelo = PINNPoderosa(in_dim=2, out_dim=3)
    x = torch.randn(100, 2)
    y = modelo(x)
    print(f"Saída shape: {y.shape}")
    print(f"Parâmetros treináveis: {modelo.num_parametros():,}")
