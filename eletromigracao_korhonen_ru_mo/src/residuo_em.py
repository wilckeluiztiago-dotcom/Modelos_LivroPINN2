"""
Resíduos Korhonen + Ohm (PyTorch autograd).
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_em import RedePotencial, RedeTensao
from .fisica_korhonen import parametros_korhonen_default


def residuo_potencial(
    rede_phi: RedePotencial,
    x: torch.Tensor,
    p: Dict[str, float] = None,
) -> torch.Tensor:
    if p is None:
        p = parametros_korhonen_default()
    phi = rede_phi(x)
    dphi = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True)[0]
    d2phi = torch.autograd.grad(dphi, x, torch.ones_like(dphi), create_graph=True)[0]
    return p["sigma_cond"] * d2phi


def residuo_korhonen(
    rede_sigma: RedeTensao,
    rede_phi: RedePotencial,
    xt: torch.Tensor,
    p: Dict[str, float] = None,
) -> torch.Tensor:
    if p is None:
        p = parametros_korhonen_default()
    sigma = rede_sigma(xt)
    g = torch.autograd.grad(sigma, xt, torch.ones_like(sigma), create_graph=True)[0]
    ds_dx = g[:, 0:1]
    ds_dt = g[:, 1:2]
    # segunda derivada em x: grad de ds_dx w.r.t. xt, pegar componente x
    g2 = torch.autograd.grad(ds_dx, xt, torch.ones_like(ds_dx), create_graph=True)[0]
    d2s_dx2 = g2[:, 0:1]

    x = xt[:, 0:1]
    # φ e derivadas — x precisa estar no grafo; usamos xt[:,0:1] que já tem grad
    phi = rede_phi(x)
    dphi = torch.autograd.grad(phi, x, torch.ones_like(phi), create_graph=True, allow_unused=True)[0]
    if dphi is None:
        dphi = torch.zeros_like(x)
    d2phi = torch.autograd.grad(dphi, x, torch.ones_like(dphi), create_graph=True, allow_unused=True)[0]
    if d2phi is None:
        d2phi = torch.zeros_like(x)

    D = p["D_eff"]
    alpha = p["Z_star_e_Omega"]
    return ds_dt - D * (d2s_dx2 + alpha * d2phi)


def residuo_fluxo_bc(
    rede_sigma: RedeTensao,
    rede_phi: RedePotencial,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    p: Dict[str, float] = None,
) -> torch.Tensor:
    if p is None:
        p = parametros_korhonen_default()
    xt = torch.cat([x_bc, t_bc], dim=1)
    sigma = rede_sigma(xt)
    g = torch.autograd.grad(sigma, xt, torch.ones_like(sigma), create_graph=True)[0]
    ds_dx = g[:, 0:1]
    phi = rede_phi(x_bc)
    dphi = torch.autograd.grad(phi, x_bc, torch.ones_like(phi), create_graph=True, allow_unused=True)[0]
    if dphi is None:
        dphi = torch.zeros_like(x_bc)
    return ds_dx + p["Z_star_e_Omega"] * dphi


def perda_em(
    rede_sigma: RedeTensao,
    rede_phi: RedePotencial,
    x_phi: torch.Tensor,
    xt_col: torch.Tensor,
    x_bc: torch.Tensor,
    t_bc: torch.Tensor,
    phi_bc_val: torch.Tensor,
    sigma0_xt: torch.Tensor,
    sigma0_val: torch.Tensor,
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_bc: float = 8.0,
    peso_ic: float = 10.0,
    peso_flux: float = 5.0,
) -> Tuple[torch.Tensor, dict]:
    R_phi = residuo_potencial(rede_phi, x_phi, p)
    R_k = residuo_korhonen(rede_sigma, rede_phi, xt_col, p)
    R_flux = residuo_fluxo_bc(rede_sigma, rede_phi, x_bc, t_bc, p)

    perda_pde = torch.mean(R_phi ** 2) + torch.mean(R_k ** 2)
    perda_flux = torch.mean(R_flux ** 2)

    # BC φ nos extremos
    x_ends = torch.tensor([[0.0], [1.0]], device=x_phi.device, dtype=x_phi.dtype)
    phi_pred = rede_phi(x_ends)
    perda_bc_phi = torch.mean((phi_pred - phi_bc_val) ** 2)

    sig0_pred = rede_sigma(sigma0_xt)
    perda_ic = torch.mean((sig0_pred - sigma0_val) ** 2)

    total = (
        peso_pde * perda_pde
        + peso_flux * perda_flux
        + peso_bc * perda_bc_phi
        + peso_ic * perda_ic
    )
    return total, {
        "pde": float(perda_pde.detach()),
        "flux": float(perda_flux.detach()),
        "bc": float(perda_bc_phi.detach()),
        "ic": float(perda_ic.detach()),
    }
