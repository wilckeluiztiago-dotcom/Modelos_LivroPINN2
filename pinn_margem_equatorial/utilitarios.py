"""
utilitarios.py
Funções auxiliares, arquiteturas de rede e perdas para PINNs.
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Callable, Optional

class RedePINN(nn.Module):
    """Rede neural fully-connected padrão para PINNs."""
    def __init__(
        self,
        camadas: List[int] = [2, 64, 64, 64, 1],
        ativacao: Callable = nn.Tanh,
        inicializacao: str = "xavier"
    ):
        super().__init__()
        self.camadas = nn.ModuleList()
        for i in range(len(camadas) - 1):
            self.camadas.append(nn.Linear(camadas[i], camadas[i + 1]))
        self.ativacao = ativacao()
        if inicializacao == "xavier":
            self._inicializar_xavier()

    def _inicializar_xavier(self):
        for camada in self.camadas:
            if isinstance(camada, nn.Linear):
                nn.init.xavier_normal_(camada.weight)
                nn.init.zeros_(camada.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, camada in enumerate(self.camadas[:-1]):
            x = self.ativacao(camada(x))
        return self.camadas[-1](x)


class RedePINNComSaidaMultipla(nn.Module):
    """Rede com múltiplas saídas (ex: pressão + saturação)."""
    def __init__(self, camadas: List[int], n_saidas: int = 2):
        super().__init__()
        self.backbone = RedePINN(camadas[:-1] + [camadas[-1]])
        self.saida = nn.Linear(camadas[-1], n_saidas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.saida(h)


def residual_mse(residuo: torch.Tensor) -> torch.Tensor:
    """Erro quadrático médio do residual físico."""
    return torch.mean(residuo ** 2)


def residual_mae(residuo: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(residuo))


def amostrar_dominio(
    n_pontos: int,
    limites: List[tuple],
    device: str = "cpu"
) -> torch.Tensor:
    """Amostra pontos uniformemente no domínio hipercúbico."""
    pontos = []
    for low, high in limites:
        pontos.append(torch.rand(n_pontos, 1, device=device) * (high - low) + low)
    return torch.cat(pontos, dim=1)


def amostrar_borda(
    n_pontos: int,
    dim_fix: int,
    valor_fix: float,
    limites_outras: List[tuple],
    device: str = "cpu"
) -> torch.Tensor:
    """Amostra pontos em uma face do domínio (condição de contorno)."""
    pontos = []
    for i, (low, high) in enumerate(limites_outras):
        if i == dim_fix:
            pontos.append(torch.full((n_pontos, 1), valor_fix, device=device))
        else:
            pontos.append(torch.rand(n_pontos, 1, device=device) * (high - low) + low)
    return torch.cat(pontos, dim=1)


def gradiente(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Calcula gradiente via autograd."""
    return torch.autograd.grad(
        y, x, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )[0]


def segunda_derivada(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Segunda derivada via autograd."""
    dy = gradiente(y, x)
    return gradiente(dy, x)


class PerdaMultiobjetivo:
    """Gerenciador de pesos de perda física + dados + contorno."""
    def __init__(self, pesos: Optional[dict] = None):
        self.pesos = pesos or {
            "fisica": 1.0,
            "dados": 10.0,
            "contorno": 10.0,
            "inicial": 10.0
        }

    def __call__(self, perdas: dict) -> torch.Tensor:
        total = 0.0
        for chave, valor in perdas.items():
            peso = self.pesos.get(chave, 1.0)
            total = total + peso * valor
        return total


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_disponivel() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
