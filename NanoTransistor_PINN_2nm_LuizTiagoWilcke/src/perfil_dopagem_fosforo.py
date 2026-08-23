"""
Módulo: Perfil de Dopagem de Fósforo realista (erfc + gaussiano)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
from parametros_materiais_si import ParametrosSilicio
from geometria_dispositivo import GeometriaNanotransistor


class PerfilDopagemFosforo(nn.Module):
    """
    Perfil realista de junção fonte/dreno com fósforo:
    - região S/D altamente dopada (2e20 cm⁻³)
    - canal levemente dopado (1e15 cm⁻³)
    - transição suave controlada por sigma (nm)
    Parâmetros podem ser liberados para calibração inversa.
    """
    def __init__(self, geo: GeometriaNanotransistor = None,
                 mat: ParametrosSilicio = None, treinavel: bool = True):
        super().__init__()
        self.geo = geo or GeometriaNanotransistor()
        self.mat = mat or ParametrosSilicio()
        self.N_ref = self.mat.N_D_SD

        if treinavel:
            self.log_N_SD = nn.Parameter(torch.tensor(torch.log(torch.tensor(self.mat.N_D_SD))))
            self.log_N_canal = nn.Parameter(torch.tensor(torch.log(torch.tensor(self.mat.N_D_canal))))
            self.sigma_nm = nn.Parameter(torch.tensor(2.2))
        else:
            self.register_buffer("log_N_SD", torch.log(torch.tensor(self.mat.N_D_SD)))
            self.register_buffer("log_N_canal", torch.log(torch.tensor(self.mat.N_D_canal)))
            self.register_buffer("sigma_nm", torch.tensor(2.2))

    @property
    def N_SD(self):
        return torch.exp(self.log_N_SD)

    @property
    def N_canal(self):
        return torch.exp(self.log_N_canal)

    def forward(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Retorna N_D em m⁻³."""
        L = self.geo.comprimento_canal_nm
        x_nm = x_norm * L
        # perfil gaussiano centrado nas extremidades
        s2 = 2.0 * self.sigma_nm**2 + 1e-8
        f_s = torch.exp(-(x_nm ** 2) / s2)
        f_d = torch.exp(-((x_nm - L) ** 2) / s2)
        N = self.N_canal + (self.N_SD - self.N_canal) * (f_s + f_d)
        return torch.clamp(N, 1e18, 5e26)

    def normalizado(self, x_norm: torch.Tensor) -> torch.Tensor:
        """N_D / N_ref (adimensional)."""
        return self.forward(x_norm) / self.N_ref

    def concentracao_cm3(self, x_norm: torch.Tensor) -> torch.Tensor:
        return self.forward(x_norm) * 1e-6
