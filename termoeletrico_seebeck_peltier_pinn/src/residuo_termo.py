"""
Resíduos PINN do transporte termoelétrico acoplado.

  J = -σ ∂x φ - σ S ∂x T
  ∇·J = 0  →  ∂x J = 0

  Π = S T
  q = Π J - κ ∂x T
  ∇·q = J · E - ∂x (Π J)     com E = -∂x φ

Equivalente (1D):
  ∂x q - J*(-∂x φ) + ∂x(Π J) = 0
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_termo import RedePINN_Termo
from .fisica_termo import parametros_termo_default


def derivadas_1d(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """∂u/∂x via autograd."""
    return torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True,
    )[0]


def residuos_termo(
    rede: RedePINN_Termo,
    x: torch.Tensor,
    p: Dict[str, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_termo_default()
    sigma, S, kappa = p["sigma"], p["S"], p["kappa"]

    phi, T = rede.campos(x)
    dphi = derivadas_1d(phi, x)
    dT = derivadas_1d(T, x)

    # corrente
    J = -sigma * dphi - sigma * S * dT
    dJ = derivadas_1d(J, x)

    # conservação de carga
    R_carga = dJ

    # Peltier + calor
    Pi = S * T
    E = -dphi
    q = Pi * J - kappa * dT
    dq = derivadas_1d(q, x)
    d_PiJ = derivadas_1d(Pi * J, x)

    # ∇·q = J·E - ∇·(Π J)
    R_energia = dq - (J * E - d_PiJ)

    return R_carga, R_energia


def perda_termo(
    rede: RedePINN_Termo,
    x_col: torch.Tensor,
    x_bc: torch.Tensor,
    phi_bc: torch.Tensor,
    T_bc: torch.Tensor,
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_bc: float = 10.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    R_c, R_e = residuos_termo(rede, x_col, p)
    perda_pde = torch.mean(R_c ** 2) + torch.mean(R_e ** 2)

    phi_pred, T_pred = rede.campos(x_bc)
    perda_bc = torch.mean((phi_pred - phi_bc) ** 2) + torch.mean((T_pred - T_bc) ** 2)

    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, perda_pde.detach(), perda_bc.detach()
