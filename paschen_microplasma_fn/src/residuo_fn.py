"""
Resíduos:

  ∇·(ε ∇φ) + q (n_i − n_e) = 0

  ∂t n_e + ∇·(−μ_e ∇φ n_e − D_e ∇ n_e)
    − α |μ_e ∇φ| n_e − G_FN(∇φ) δ_catodo ≈ 0

G_FN aplicado suavemente perto do cátodo (x≈0).
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_fn import RedePINN_FN
from .fisica_paschen import parametros_fn_default


def residuos_fn(
    rede: RedePINN_FN,
    xt: torch.Tensor,
    p: Dict[str, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_fn_default()
    phi, ne = rede.campos(xt)
    x = xt[:, 0:1]

    def g1(u, coord_slice):
        return torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0][:, coord_slice]

    dphi_dx = g1(phi, slice(0, 1))
    dphi_dt = g1(phi, slice(1, 2))
    dne_dx = g1(ne, slice(0, 1))
    dne_dt = g1(ne, slice(1, 2))
    d2phi = torch.autograd.grad(dphi_dx, xt, torch.ones_like(dphi_dx), create_graph=True)[0][:, 0:1]
    d2ne = torch.autograd.grad(dne_dx, xt, torch.ones_like(dne_dx), create_graph=True)[0][:, 0:1]

    # Poisson
    n_i = p["n_i0"]
    R_poisson = p["eps"] * d2phi + p["q"] * (n_i - ne)

    # fluxo de elétrons: v = −μ E = μ ∇φ  (E=−∇φ)
    # Γ = −μ_e (∂x φ) n_e − D_e ∂x n_e
    Gamma = -p["mu_e"] * dphi_dx * ne - p["D_e"] * dne_dx
    dGamma_dx = torch.autograd.grad(Gamma, xt, torch.ones_like(Gamma), create_graph=True)[0][:, 0:1]

    E_abs = torch.abs(dphi_dx) + 1e-6
    ioniz = p["alpha0"] * p["mu_e"] * E_abs * ne

    # FN suavizado no cátodo x≈0
    A, B = p["A_FN"], p["B_FN"]
    Gfn = A * E_abs ** 2 * torch.exp(-B / E_abs)
    weight_cat = torch.exp(-((x - 0.0) / 0.05) ** 2)  # pico no cátodo

    R_ne = dne_dt + dGamma_dx - ioniz - Gfn * weight_cat
    return R_poisson, R_ne


def perda_fn(
    rede: RedePINN_FN,
    xt_col: torch.Tensor,
    xt_bc: torch.Tensor,
    phi_bc: torch.Tensor,
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_bc: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    Rp, Rn = residuos_fn(rede, xt_col, p)
    perda_pde = torch.mean(Rp ** 2) + torch.mean(Rn ** 2)
    phi_b, _ = rede.campos(xt_bc)
    perda_bc = torch.mean((phi_b - phi_bc) ** 2)
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, {"pde": float(perda_pde.detach()), "bc": float(perda_bc.detach())}
