"""
Módulo 12: Otimizadores Híbridos (Adam + L-BFGS)
Autor: Luiz Tiago Wilcke
"""

import torch
from torch.optim import Adam, LBFGS
from .config import CONFIG
from .arquitetura_deeponet import PIDeepONetHJM


def criar_otimizador_adam(modelo: PIDeepONetHJM, lr: float = CONFIG.taxa_aprendizado) -> Adam:
    return Adam(modelo.parameters(), lr=lr, betas=(0.9, 0.999))


def criar_otimizador_lbfgs(modelo: PIDeepONetHJM, lr: float = 1.0) -> LBFGS:
    return LBFGS(
        modelo.parameters(),
        lr=lr,
        max_iter=20,
        history_size=50,
        line_search_fn="strong_wolfe",
    )


def passo_lbfgs(otimizador: LBFGS, closure) -> float:
    """Executa um passo L-BFGS e retorna a perda."""
    loss = otimizador.step(closure)
    return loss.item() if torch.is_tensor(loss) else loss
