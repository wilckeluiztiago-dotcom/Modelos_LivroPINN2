"""
Módulo 07: Arquitetura PI-DeepONet Completa
Autor: Luiz Tiago Wilcke

P(t, T) = exp( Σ_k b_k(u) · t_k(y) + b0 )
"""

import torch
import torch.nn as nn
from .config import CONFIG
from .rede_branch import RedeBranch
from .rede_trunk import RedeTrunk


class PIDeepONetHJM(nn.Module):
    """
    Physics-Informed Deep Operator Network para o modelo HJM.

    Entrada:
        u : (batch, m)  – curva forward inicial nos sensores
        t : (batch, 1)  – tempo atual
        T : (batch, 1)  – maturidade do título

    Saída:
        P : (batch, 1)  – preço do título zero-cupom P(t, T)
    """

    def __init__(self, config=CONFIG):
        super().__init__()
        self.config = config
        self.branch = RedeBranch(
            dim_entrada=config.num_sensores,
            camadas_ocultas=config.camadas_branch,
            dim_saida=config.dim_latent,
            ativacao=config.ativacao,
        )
        self.trunk = RedeTrunk(
            dim_entrada=2,
            camadas_ocultas=config.camadas_trunk,
            dim_saida=config.dim_latent,
            ativacao=config.ativacao,
        )
        # Bias escalar aprendível
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, u: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Computa P(t,T) = exp( <b(u), t(y)> + b0 )
        """
        y = torch.cat([t, T], dim=-1)  # (batch, 2)
        b = self.branch(u)             # (batch, p)
        tr = self.trunk(y)             # (batch, p)
        produto = torch.sum(b * tr, dim=-1, keepdim=True) + self.bias
        # Garantia de positividade e estabilidade numérica
        P = torch.exp(produto)
        return torch.clamp(P, min=1e-8, max=2.0)

    def log_preco(self, u: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """Retorna log P para facilitar derivadas."""
        y = torch.cat([t, T], dim=-1)
        b = self.branch(u)
        tr = self.trunk(y)
        return torch.sum(b * tr, dim=-1, keepdim=True) + self.bias
