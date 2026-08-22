# -*- coding: utf-8 -*-
"""
Módulo 25: Simulador Monte Carlo Cinético (Gillespie) de Referência
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .taxas_tunelamento import taxas_tunelamento
from .configuracao_dispositivo import ConfiguracaoSET
from .constantes_fisicas import DTYPE, DEVICE

def trajetoria_gillespie(
    cfg: ConfiguracaoSET,
    V_D: float,
    V_G: float,
    n0: int = 0,
    t_max: float = 1e-6,
    n_passos: int = 5000
) -> tuple:
    """Gera trajetória de estados de carga n(t) via algoritmo de Gillespie."""
    n = torch.tensor([float(n0)], dtype=DTYPE, device=DEVICE)
    t = torch.tensor([0.0], dtype=DTYPE, device=DEVICE)
    historico_n = [n0]
    historico_t = [0.0]

    VD = torch.tensor([[V_D]], dtype=DTYPE, device=DEVICE)
    VG = torch.tensor([[V_G]], dtype=DTYPE, device=DEVICE)

    for _ in range(n_passos):
        Gs_mais, Gs_menos, Gd_mais, Gd_menos = taxas_tunelamento(n, VD, VG, cfg)
        taxas = torch.stack([Gs_mais, Gs_menos, Gd_mais, Gd_menos]).flatten()
        taxa_total = taxas.sum()
        if taxa_total <= 0:
            break
        dt = -torch.log(torch.rand(1, device=DEVICE) + 1e-20) / (taxa_total + 1e-20)
        t = t + dt
        if t.item() > t_max:
            break
        probs = taxas / (taxa_total + 1e-20)
        escolha = torch.multinomial(probs, 1).item()
        if escolha in (0, 2):
            n = n + 1
        else:
            n = n - 1
        historico_n.append(int(n.item()))
        historico_t.append(t.item())

    return torch.tensor(historico_t, dtype=DTYPE), torch.tensor(historico_n, dtype=DTYPE)
