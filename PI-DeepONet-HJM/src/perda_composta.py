"""
Módulo 10: Função de Perda Composta (Física + Dados + Livre-Arbitragem)
Autor: Luiz Tiago Wilcke
"""

import torch
from .config import CONFIG
from .residual_hjm import residual_livre_arbitragem
from .residual_pde import residual_edp_titulo, residual_condicao_inicial
from .arquitetura_deeponet import PIDeepONetHJM


def perda_composta(
    modelo: PIDeepONetHJM,
    batch: dict,
    T_sensores: torch.Tensor,
    peso_fisica: float = CONFIG.peso_fisica,
    peso_dados: float = CONFIG.peso_dados,
    peso_la: float = CONFIG.peso_livre_arbitragem,
) -> dict:
    """
    Perda total = λ_fis * ||R_EDP||² + λ_dados * ||P(0,T) - P0||² + λ_LA * ||R_HJM||²
    """
    u = batch["u"]
    t = batch["t"]
    T = batch["T"]

    # Residual da EDP
    R_edp = residual_edp_titulo(modelo, u, t, T)
    loss_fisica = torch.mean(R_edp ** 2)

    # Condição inicial (dados)
    R_ci = residual_condicao_inicial(modelo, u, T, T_sensores)
    loss_dados = torch.mean(R_ci ** 2)

    # Residual de livre-arbitragem (regularização)
    R_la = residual_livre_arbitragem(modelo, u, t, T)
    loss_la = torch.mean(R_la ** 2) if R_la is not None else torch.tensor(0.0, device=u.device)

    loss_total = (
        peso_fisica * loss_fisica
        + peso_dados * loss_dados
        + peso_la * loss_la
    )

    return {
        "total": loss_total,
        "fisica": loss_fisica.detach(),
        "dados": loss_dados.detach(),
        "livre_arbitragem": loss_la.detach() if isinstance(loss_la, torch.Tensor) else loss_la,
    }
