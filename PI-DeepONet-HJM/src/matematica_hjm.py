"""
Módulo 03: Matemática do Modelo Heath-Jarrow-Morton (HJM)
Autor: Luiz Tiago Wilcke

Implementa a condição de livre-arbitragem e relações fundamentais.
"""

import torch
from typing import Callable, Optional
from .config import CONFIG
from .utils import para_tensor


def volatilidade_hjm(
    t: torch.Tensor,
    T: torch.Tensor,
    tipo: str = CONFIG.tipo_volatilidade,
    sigma0: float = CONFIG.sigma_constante,
    kappa: float = 0.1,
) -> torch.Tensor:
    """
    Função de volatilidade σ(t, T) do modelo HJM.

    Tipos:
      - constante: σ(t,T) = σ0
      - exponencial: σ(t,T) = σ0 * exp(-κ(T-t))
      - hull_white: σ(t,T) = σ0 * (1 - exp(-κ(T-t)))/κ
    """
    tau = torch.clamp(T - t, min=0.0)
    if tipo == "constante":
        return torch.full_like(t, sigma0)
    elif tipo == "exponencial":
        return sigma0 * torch.exp(-kappa * tau)
    elif tipo == "hull_white":
        return sigma0 * (1.0 - torch.exp(-kappa * tau)) / (kappa + 1e-8)
    else:
        raise ValueError(f"Tipo de volatilidade desconhecido: {tipo}")


def drift_livre_arbitragem(
    t: torch.Tensor,
    T: torch.Tensor,
    sigma_func: Callable = volatilidade_hjm,
    num_pontos_integracao: int = 64,
) -> torch.Tensor:
    """
    Condição de livre-arbitragem de Heath-Jarrow-Morton:

        α(t, T) = σ(t, T) * ∫_t^T σ(t, s) ds

    Calculado por quadratura de Gauss-Legendre em [t, T].
    """
    # Pontos de Gauss-Legendre em [-1, 1]
    nos, pesos = torch.tensor(
        [
            [-0.9061798459, 0.2369268850],
            [-0.5384693101, 0.4786286705],
            [0.0, 0.5688888889],
            [0.5384693101, 0.4786286705],
            [0.9061798459, 0.2369268850],
        ],
        dtype=CONFIG.dtype,
        device=t.device,
    ).T  # simplificado; para produção use scipy ou mais nós

    # Mapeia [t, T] -> [-1, 1]
    a = t
    b = T
    meio = 0.5 * (b - a)
    centro = 0.5 * (b + a)

    integral = torch.zeros_like(t)
    for i in range(len(nos)):
        s = centro + meio * nos[i]
        sigma_s = sigma_func(t, s)
        integral = integral + pesos[i] * sigma_s

    integral = integral * meio
    sigma_T = sigma_func(t, T)
    alpha = sigma_T * integral
    return alpha


def volatilidade_preco_titulo(
    t: torch.Tensor,
    T: torch.Tensor,
    sigma_func: Callable = volatilidade_hjm,
    num_pontos: int = 32,
) -> torch.Tensor:
    """
    Volatilidade do preço do título zero-cupom:

        σ_P(t, T) = - ∫_t^T σ(t, s) ds

    (sinal negativo por convenção de duration).
    """
    a = t
    b = T
    meio = 0.5 * (b - a)
    centro = 0.5 * (b + a)

    nos = torch.linspace(-1.0, 1.0, num_pontos, device=t.device, dtype=CONFIG.dtype)
    pesos = torch.ones_like(nos) * (2.0 / num_pontos)

    integral = torch.zeros_like(t)
    for i in range(num_pontos):
        s = centro + meio * nos[i]
        integral = integral + pesos[i] * sigma_func(t, s)

    sigma_P = -integral * meio
    return sigma_P


def relacao_preco_forward(P: torch.Tensor, t: torch.Tensor, T: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relação fundamental: f(t, T) = - ∂_T log P(t, T)
    """
    log_P = torch.log(torch.clamp(P, min=eps))
    # Diferença finita centrada aproximada (em produção use autograd)
    dT = 1e-4
    # Assume que o chamador fornecerá gradiente via autograd
    return -torch.autograd.grad(log_P.sum(), T, create_graph=True)[0]


def taxa_instantanea(P: torch.Tensor, t: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """r(t) = f(t, t) = - ∂_t log P(t, t) quando T=t."""
    log_P = torch.log(torch.clamp(P, min=1e-8))
    return -torch.autograd.grad(log_P.sum(), t, create_graph=True)[0]
