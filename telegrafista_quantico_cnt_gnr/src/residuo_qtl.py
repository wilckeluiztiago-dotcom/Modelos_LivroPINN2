"""
Resíduos do telegrafista quântico:

  ∂V/∂z + R I + L_tot ∂I/∂t = 0
  ∂I/∂z + G V + C_eff ∂V/∂t = 0

  BC terminal: V(L,t) − Z_L I(L,t) = 0
  BC fonte:    V(0,t) = V_src(t)
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_qtl import RedePINN_QTL
from .fisica_telegrafista import parametros_qtl_default


def residuos_qtl(
    rede: RedePINN_QTL,
    zt: torch.Tensor,
    p: Dict[str, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_qtl_default()
    V, I = rede.campos(zt)

    def grads(u):
        g = torch.autograd.grad(u, zt, torch.ones_like(u), create_graph=True)[0]
        return g[:, 0:1], g[:, 1:2]

    dV_dz, dV_dt = grads(V)
    dI_dz, dI_dt = grads(I)

    R1 = dV_dz + p["R"] * I + p["L_tot"] * dI_dt
    R2 = dI_dz + p["G"] * V + p["C_eff"] * dV_dt
    return R1, R2


def perda_qtl(
    rede: RedePINN_QTL,
    zt_col: torch.Tensor,
    zt_src: torch.Tensor,
    V_src: torch.Tensor,
    zt_load: torch.Tensor,
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_src: float = 10.0,
    peso_term: float = 5.0,
) -> Tuple[torch.Tensor, dict]:
    if p is None:
        p = parametros_qtl_default()
    R1, R2 = residuos_qtl(rede, zt_col, p)
    perda_pde = torch.mean(R1 ** 2) + torch.mean(R2 ** 2)

    V_s, _ = rede.campos(zt_src)
    perda_src = torch.mean((V_s - V_src) ** 2)

    V_L, I_L = rede.campos(zt_load)
    perda_term = torch.mean((V_L - p["Z_L"] * I_L) ** 2)

    total = peso_pde * perda_pde + peso_src * perda_src + peso_term * perda_term
    return total, {
        "pde": float(perda_pde.detach()),
        "src": float(perda_src.detach()),
        "term": float(perda_term.detach()),
    }
