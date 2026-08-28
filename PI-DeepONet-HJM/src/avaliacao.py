"""
Módulo 14: Avaliação e Métricas
Autor: Luiz Tiago Wilcke
"""

import torch
from .config import CONFIG
from .arquitetura_deeponet import PIDeepONetHJM
from .geracao_curvas import gerar_ensemble_curvas, curva_forward_nelson_siegel
from .residual_pde import residual_edp_titulo


@torch.no_grad()
def avaliar_superficie(
    modelo: PIDeepONetHJM,
    u: torch.Tensor,
    resolucao: int = 50,
) -> dict:
    """
    Avalia a superfície P(t,T) em uma grade regular.
    """
    modelo.eval()
    t_grid = torch.linspace(0, CONFIG.t_max, resolucao, device=CONFIG.dispositivo)
    T_grid = torch.linspace(0.1, CONFIG.T_max, resolucao, device=CONFIG.dispositivo)
    tt, TT = torch.meshgrid(t_grid, T_grid, indexing="ij")

    t_flat = tt.reshape(-1, 1)
    T_flat = TT.reshape(-1, 1)
    u_exp = u.expand(t_flat.shape[0], -1)

    P = modelo(u_exp, t_flat, T_flat).reshape(resolucao, resolucao)

    return {
        "t": tt.cpu().numpy(),
        "T": TT.cpu().numpy(),
        "P": P.cpu().numpy(),
    }


def erro_relativo_medio(
    modelo: PIDeepONetHJM,
    num_amostras: int = 1000,
) -> float:
    """Calcula o residual médio normalizado da EDP."""
    modelo.eval()
    from .amostragem import amostrar_dominio
    batch = amostrar_dominio(num_pontos=num_amostras)
    R = residual_edp_titulo(modelo, batch["u"], batch["t"], batch["T"])
    return torch.mean(torch.abs(R)).item()
