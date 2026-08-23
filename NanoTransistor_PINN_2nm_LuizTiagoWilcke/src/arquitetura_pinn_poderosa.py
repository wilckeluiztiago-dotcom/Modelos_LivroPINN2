"""
Módulo 09: Arquitetura PINN Poderosa (Residual MLP + Fourier Features)
Autor: Luiz Tiago Wilcke
Inspirada nas redes do livro de PINNs financeiras, adaptada para multi-física.
"""

import torch
import torch.nn as nn
import math

class FourierFeatures(nn.Module):
    def __init__(self, in_dim=1, num_freq=64, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_dim, num_freq) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = 2 * math.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, dim, activation=nn.Tanh):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = activation()

    def forward(self, x):
        residual = x
        out = self.act(self.fc1(x))
        out = self.fc2(out)
        return self.act(out + residual)

class PINNPoderosa(nn.Module):
    """
    Rede multi-saída para (φ, n, ψ, ...) 
    Entrada: coordenadas normalizadas (x, y, Vgs, Vds, ...)
    """
    def __init__(self, in_dim=2, out_dim=3, hidden=256, n_blocks=6, fourier=True):
        super().__init__()
        self.fourier = fourier
        if fourier:
            self.ff = FourierFeatures(in_dim, num_freq=64)
            first_dim = 128
        else:
            first_dim = in_dim

        layers = [nn.Linear(first_dim, hidden), nn.Tanh()]
        for _ in range(n_blocks):
            layers.append(ResidualBlock(hidden))
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)

        # inicialização Xavier
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, coords):
        if self.fourier:
            x = self.ff(coords)
        else:
            x = coords
        return self.net(x)

if __name__ == "__main__":
    modelo = PINNPoderosa(in_dim=2, out_dim=3)
    x = torch.randn(100, 2)
    y = modelo(x)
    print(f"Saída shape: {y.shape}")
    print(f"Parâmetros: {sum(p.numel() for p in modelo.parameters()):,}")
