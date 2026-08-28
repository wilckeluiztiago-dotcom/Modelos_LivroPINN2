"""
Módulo 11: Amostragem de Pontos no Domínio (Latin Hypercube / Uniforme)
Autor: Luiz Tiago Wilcke
"""

import torch
from .config import CONFIG
from .geracao_curvas import gerar_ensemble_curvas, gerar_pontos_treino


def amostrar_dominio(
    num_pontos: int = CONFIG.num_pontos_dominio,
    num_curvas: int = 8,
) -> dict:
    """
    Amostragem uniforme no domínio (t,T) com T ≥ t,
    associada a um mini-batch de curvas u.
    """
    u, T_sensores = gerar_ensemble_curvas(num_curvas=num_curvas)
    # Seleciona uma curva aleatória por ponto
    idx = torch.randint(0, num_curvas, (num_pontos,))
    u_batch = u[idx]

    t = torch.rand(num_pontos, 1, device=CONFIG.dispositivo) * CONFIG.t_max
    T = t + torch.rand(num_pontos, 1, device=CONFIG.dispositivo) * (CONFIG.T_max - t)

    return {
        "t": t.requires_grad_(True),
        "T": T.requires_grad_(True),
        "u": u_batch.to(CONFIG.dispositivo),
        "T_sensores": T_sensores.to(CONFIG.dispositivo),
    }


def amostrar_contorno_inicial(num_pontos: int = CONFIG.num_pontos_contorno) -> dict:
    """Amostra pontos em t=0 para condição inicial."""
    u, T_sensores = gerar_ensemble_curvas(num_curvas=1)
    T = torch.rand(num_pontos, 1, device=CONFIG.dispositivo) * CONFIG.T_max
    t = torch.zeros_like(T)
    u_expand = u.expand(num_pontos, -1).to(CONFIG.dispositivo)

    return {
        "t": t.requires_grad_(True),
        "T": T.requires_grad_(True),
        "u": u_expand,
        "T_sensores": T_sensores.to(CONFIG.dispositivo),
    }
