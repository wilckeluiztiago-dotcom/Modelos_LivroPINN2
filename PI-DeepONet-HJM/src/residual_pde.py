"""
Módulo 09: Residual da EDP de Precificação do Título Zero-Cupom
Autor: Luiz Tiago Wilcke

EDP:
    ∂P/∂t + r(t) P - f(t,T) P + (1/2) σ_P²(t,T) P = 0
com r(t) = f(t,t)
"""

import torch
from .config import CONFIG
from .matematica_hjm import volatilidade_hjm, volatilidade_preco_titulo
from .arquitetura_deeponet import PIDeepONetHJM


def residual_edp_titulo(
    modelo: PIDeepONetHJM,
    u: torch.Tensor,
    t: torch.Tensor,
    T: torch.Tensor,
) -> torch.Tensor:
    """
    Calcula o residual da EDP de Black-Scholes / HJM para o título:

        R = ∂P/∂t + r P - f P + ½ σ_P² P
    """
    t = t.clone().requires_grad_(True)
    T = T.clone().requires_grad_(True)

    P = modelo(u, t, T)  # (batch, 1)

    # Derivadas via autograd
    dP_dt = torch.autograd.grad(
        P, t, grad_outputs=torch.ones_like(P), create_graph=True
    )[0]

    dP_dT = torch.autograd.grad(
        P, T, grad_outputs=torch.ones_like(P), create_graph=True
    )[0]

    # f(t,T) = - (1/P) * ∂P/∂T
    f_tT = -dP_dT / (P + 1e-8)

    # r(t) ≈ f(t, t) – usamos média local quando T ≈ t
    # Para simplicidade, interpolamos ou usamos f quando T próximo
    # Em prática: r(t) = f(t,t) obtido por limite
    mascara_perto = (T - t) < 0.05
    r = torch.where(mascara_perto, f_tT, f_tT.detach() * 0.0 + 0.03)  # fallback

    # Volatilidade do preço
    sigma_P = volatilidade_preco_titulo(t, T)
    sigma_P2 = sigma_P ** 2

    # Residual da EDP
    residual = dP_dt + r * P - f_tT * P + 0.5 * sigma_P2 * P

    return residual


def residual_condicao_inicial(
    modelo: PIDeepONetHJM,
    u: torch.Tensor,
    T: torch.Tensor,
    T_sensores: torch.Tensor,
) -> torch.Tensor:
    """
    Condição inicial: P(0, T) = exp( -∫_0^T f(0,s) ds )
    """
    t0 = torch.zeros_like(T)
    P0_pred = modelo(u, t0, T)

    # Integral da curva u nos sensores (aproximação trapézios)
    dt = T_sensores[1] - T_sensores[0] if len(T_sensores) > 1 else 0.1
    # u shape (batch, m)
    integral = torch.cumsum(u, dim=-1) * dt
    # Interpola para T
    # Aproximação: usa o último valor se T grande
    idx = torch.clamp(
        ((T / (T_sensores[-1] + 1e-8)) * (len(T_sensores) - 1)).long(),
        0, len(T_sensores) - 1
    ).squeeze()
    if idx.dim() == 0:
        idx = idx.unsqueeze(0)
    integral_T = integral[torch.arange(u.shape[0]), idx]
    P0_exato = torch.exp(-integral_T).unsqueeze(-1)

    return P0_pred - P0_exato
