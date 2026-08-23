"""
Módulo: Perfil de Dopagem de Fósforo (P) realista
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
from parametros_materiais_si import ParametrosSilicio
from geometria_dispositivo import GeometriaNanotransistor


class PerfilDopagemFosforo(nn.Module):
    """
    Perfil de dopagem gaussiano/erfc para S/D + canal baixo.
    Parâmetros podem ser treináveis para calibração inversa.
    """
    def __init__(self, geo: GeometriaNanotransistor = None, mat: ParametrosSilicio = None,
                 treinavel: bool = True):
        super().__init__()
        self.geo = geo or GeometriaNanotransistor()
        self.mat = mat or ParametrosSilicio()

        if treinavel:
            self.N_SD = nn.Parameter(torch.tensor(float(self.mat.N_D_SD)))
            self.N_canal = nn.Parameter(torch.tensor(float(self.mat.N_D_canal)))
            self.sigma_nm = nn.Parameter(torch.tensor(2.5))
        else:
            self.register_buffer("N_SD", torch.tensor(float(self.mat.N_D_SD)))
            self.register_buffer("N_canal", torch.tensor(float(self.mat.N_D_canal)))
            self.register_buffer("sigma_nm", torch.tensor(2.5))

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """
        x_norm ∈ [0,1] ao longo do canal.
        Retorna N_D(x) em m⁻³.
        """
        L = self.geo.comprimento_canal_nm
        x_nm = x_norm * L
        # perfil alto nas extremidades (S/D), baixo no canal
        perfil_fonte = torch.exp(-((x_nm - 0.0)**2) / (2.0 * self.sigma_nm**2))
        perfil_dreno = torch.exp(-((x_nm - L)**2) / (2.0 * self.sigma_nm**2))
        N = self.N_canal + (self.N_SD - self.N_canal) * (perfil_fonte + perfil_dreno)
        return torch.clamp(N, min=1e18, max=5e26)

    def ionizado(self, x_norm: torch.Tensor, phi: torch.Tensor = None, T: float = 300.0) -> torch.Tensor:
        """Aproximação: fósforo em Si ~100% ionizado à temperatura ambiente."""
        return self.forward(x_norm)

    def concentracao_cm3(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.forward(x_norm) * 1e-6


if __name__ == "__main__":
    perfil = PerfilDopagemFosforo()
    x = torch.linspace(0, 1, 100)
    Nd = perfil(x)
    print(f"N_D min/max: {Nd.min().item():.2e} / {Nd.max().item():.2e} m⁻³")
    print(f"N_D em cm⁻³: {perfil.concentracao_cm3(x).max().item():.2e}")
