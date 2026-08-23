"""
Módulo 03: Perfil de Dopagem de Fósforo (P)
Autor: Luiz Tiago Wilcke
Modelo realista: Gaussiano / erfc para S/D e canal baixo.
"""

import torch
import torch.nn as nn
from parametros_materiais_si import ParametrosSilicio
from geometria_dispositivo import GeometriaNanotransistor

class PerfilDopagemFosforo(nn.Module):
    def __init__(self, geo: GeometriaNanotransistor = None, mat: ParametrosSilicio = None):
        super().__init__()
        self.geo = geo or GeometriaNanotransistor()
        self.mat = mat or ParametrosSilicio()
        # parâmetros treináveis opcionais para calibração inversa
        self.N_SD = nn.Parameter(torch.tensor(self.mat.N_D_SD, dtype=torch.float32))
        self.N_canal = nn.Parameter(torch.tensor(self.mat.N_D_canal, dtype=torch.float32))
        self.sigma_nm = nn.Parameter(torch.tensor(3.0))  # largura da junção ~ nm

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm ∈ [0,1] ao longo do canal.
        Retorna N_D(x) em m⁻³.
        """
        L = self.geo.comprimento_canal_nm
        x_nm = x_norm * L
        # posições centrais das regiões S/D (aprox. 0 e L)
        # perfil: alto nas extremidades, baixo no meio
        perfil_fonte = torch.exp( -((x_nm - 0.0)**2) / (2 * self.sigma_nm**2) )
        perfil_dreno = torch.exp( -((x_nm - L)**2) / (2 * self.sigma_nm**2) )
        N = self.N_canal + (self.N_SD - self.N_canal) * (perfil_fonte + perfil_dreno)
        # limitar máximo físico
        N = torch.clamp(N, max=5e26)
        return N

    def ionizado(self, x_norm, phi, T=300.0):
        """Aproximação de ionização completa para fósforo em Si a 300 K."""
        return self.forward(x_norm)  # para P em Si, ~100% ionizado à RT

if __name__ == "__main__":
    perfil = PerfilDopagemFosforo()
    x = torch.linspace(0, 1, 100)
    Nd = perfil(x)
    print(f"N_D min/max: {Nd.min():.2e} / {Nd.max():.2e} m⁻³")
