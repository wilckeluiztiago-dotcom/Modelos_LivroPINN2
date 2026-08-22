"""
Resíduos estacionários 1D:

  d/dx [ −q μ_n n dφ/dx + q D_n dn/dx ] − q G + q R = 0
  d/dx [ −q μ_p p dφ/dx − q D_p dp/dx ] + q G − q R = 0
  d/dx (ε dφ/dx) + q (p − n + N_net) = 0
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_btbt import RedePINN_BTBT
from .fisica_btbt import parametros_btbt_default


def residuos_btbt(
    rede: RedePINN_BTBT,
    x: torch.Tensor,
    p: Dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_btbt_default()
    phi, n, p_h = rede.campos(x)

    def dx(u):
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    dphi = dx(phi)
    dn, dp = dx(n), dx(p_h)
    d2phi = dx(dphi)

    E_abs = torch.abs(dphi) + 1e-4
    Eg = p["E_g"]
    G = p["A_Kane"] * E_abs ** 2 / (Eg ** 0.5) * torch.exp(-p["B_Kane"] * Eg ** 1.5 / E_abs)

    ni = p["n_i"]
    R = (n * p_h - ni ** 2) / (p["tau_R"] * (n + p_h + 2 * ni) + 1e-8)

    # dopagem TFET
    L = p["L"]
    N_net = torch.where(
        x < 0.3 * L,
        torch.full_like(x, -p["N_A"]),
        torch.where(x > 0.7 * L, torch.full_like(x, p["N_D"]), torch.zeros_like(x)),
    )

    Jn = -p["q"] * p["mu_n"] * n * dphi + p["q"] * p["D_n"] * dn
    Jp = -p["q"] * p["mu_p"] * p_h * dphi - p["q"] * p["D_p"] * dp
    dJn, dJp = dx(Jn), dx(Jp)

    R_n = dJn - p["q"] * G + p["q"] * R
    R_p = dJp + p["q"] * G - p["q"] * R
    R_poi = p["eps"] * d2phi + p["q"] * (p_h - n + N_net)
    return R_n, R_p, R_poi


def perda_btbt(
    rede: RedePINN_BTBT,
    x_col: torch.Tensor,
    x_bc: torch.Tensor,
    phi_bc: torch.Tensor,
    p: Dict = None,
    peso_pde: float = 1.0,
    peso_bc: float = 8.0,
) -> Tuple[torch.Tensor, dict]:
    Rn, Rp, Rpoi = residuos_btbt(rede, x_col, p)
    perda_pde = torch.mean(Rn ** 2) + torch.mean(Rp ** 2) + torch.mean(Rpoi ** 2)
    phi_b, _, _ = rede.campos(x_bc)
    perda_bc = torch.mean((phi_b - phi_bc) ** 2)
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, {
        "pde": float(perda_pde.detach()),
        "bc": float(perda_bc.detach()),
    }
