"""
Módulo 04: Geração de Curvas Forward Iniciais e Dados Sintéticos
Autor: Luiz Tiago Wilcke
"""

import torch
import numpy as np
from typing import Tuple, List
from .config import CONFIG
from .utils import para_tensor, definir_semente
from .matematica_hjm import volatilidade_hjm, drift_livre_arbitragem


def curva_forward_nelson_siegel(
    T: torch.Tensor,
    beta0: float = 0.03,
    beta1: float = -0.01,
    beta2: float = 0.02,
    tau: float = 1.5,
) -> torch.Tensor:
    """
    Curva forward inicial estilo Nelson-Siegel:

        f(0, T) = β0 + β1 * exp(-T/τ) + β2 * (T/τ) * exp(-T/τ)
    """
    x = T / tau
    return beta0 + beta1 * torch.exp(-x) + beta2 * x * torch.exp(-x)


def gerar_ensemble_curvas(
    num_curvas: int = 32,
    num_sensores: int = CONFIG.num_sensores,
    T_max: float = CONFIG.T_max,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Gera um ensemble de curvas forward iniciais f(0, ·) amostradas
    em m sensores, com parâmetros Nelson-Siegel aleatórios.

    Retorna:
        u : (num_curvas, m)  – valores da curva nos sensores
        T_sensores : (m,)    – maturidades dos sensores
    """
    definir_semente()
    T_sensores = para_tensor(CONFIG.maturidades_sensores)
    u_lista = []

    for _ in range(num_curvas):
        b0 = np.random.uniform(0.01, 0.06)
        b1 = np.random.uniform(-0.03, 0.01)
        b2 = np.random.uniform(-0.02, 0.04)
        tau = np.random.uniform(0.5, 3.0)
        f = curva_forward_nelson_siegel(T_sensores, b0, b1, b2, tau)
        u_lista.append(f)

    u = torch.stack(u_lista, dim=0)  # (N, m)
    return u, T_sensores


def preco_titulo_analitico_aproximado(
    t: torch.Tensor,
    T: torch.Tensor,
    f0: torch.Tensor,
    T_sensores: torch.Tensor,
) -> torch.Tensor:
    """
    Aproximação do preço P(t,T) sob HJM com volatilidade constante
    e curva inicial interpolada linearmente.
    Usado apenas para gerar dados de supervisão (quando disponíveis).
    """
    # Interpola f(0, ·) para obter integral aproximada
    # P(0,T) = exp( -∫_0^T f(0,s) ds )
    # Para t>0 usa a dinâmica HJM simplificada (exemplo didático)
    from torch.nn.functional import interpolate

    # Integral simples por trapézios
    dt = T_sensores[1] - T_sensores[0]
    integral = torch.cumsum(f0, dim=-1) * dt
    # Extensão para T arbitrário (interpolação linear)
    idx = torch.searchsorted(T_sensores, T.clamp(max=T_sensores[-1]))
    idx = idx.clamp(1, len(T_sensores) - 1)
    # Aproximação grosseira
    P0 = torch.exp(-integral[..., idx])
    # Ajuste temporal simples
    r0 = f0[..., 0]
    P = P0 * torch.exp(-r0 * t)
    return torch.clamp(P, min=1e-6, max=1.0)


def gerar_pontos_treino(
    num_pontos: int = CONFIG.num_pontos_dominio,
    u: torch.Tensor = None,
) -> dict:
    """
    Gera pontos de colocation (t, T) no domínio [0,t_max] x [0,T_max]
    e associa a uma curva u aleatória do ensemble.
    """
    if u is None:
        u, _ = gerar_ensemble_curvas(num_curvas=1)

    t = torch.rand(num_pontos, 1, device=CONFIG.dispositivo) * CONFIG.t_max
    T = torch.rand(num_pontos, 1, device=CONFIG.dispositivo) * CONFIG.T_max
    # Garante T >= t
    T = torch.maximum(T, t + 1e-3)

    # Repete a curva u para todos os pontos
    u_expand = u.expand(num_pontos, -1)

    return {
        "t": t.requires_grad_(True),
        "T": T.requires_grad_(True),
        "u": u_expand,
    }
