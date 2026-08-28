"""
Módulo 05: Rede Branch (Processa a curva forward inicial)
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
from typing import List
from .config import CONFIG


class RedeBranch(nn.Module):
    """
    Branch Net: mapeia a curva forward u = [f(0,T1), ..., f(0,Tm)]
    para o vetor de coeficientes b(u) ∈ R^p.
    """

    def __init__(
        self,
        dim_entrada: int = CONFIG.num_sensores,
        camadas_ocultas: List[int] = None,
        dim_saida: int = CONFIG.dim_latent,
        ativacao: str = CONFIG.ativacao,
    ):
        super().__init__()
        if camadas_ocultas is None:
            camadas_ocultas = CONFIG.camadas_branch

        camadas = []
        dims = [dim_entrada] + camadas_ocultas + [dim_saida]
        for i in range(len(dims) - 1):
            camadas.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                if ativacao == "tanh":
                    camadas.append(nn.Tanh())
                elif ativacao == "silu":
                    camadas.append(nn.SiLU())
                elif ativacao == "gelu":
                    camadas.append(nn.GELU())
                else:
                    camadas.append(nn.Tanh())

        self.rede = nn.Sequential(*camadas)
        self._inicializar_pesos()

    def _inicializar_pesos(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u : (batch, m) -> b(u) : (batch, p)
        """
        return self.rede(u)
