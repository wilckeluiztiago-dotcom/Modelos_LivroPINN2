"""
Resíduo Schrödinger + BC de onda aberta:

  H ψ = E ψ
  em x=0: ψ' − i k_L ψ ≈ 2 i k_L A_inc  (incidência da esquerda)
  em x=L: ψ' + i k_R ψ ≈ 0              (só onda transmitida)
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .rede_pinn_dbrtd import RedePINN_DBRTD
from .fisica_dbrtd import parametros_dbrtd_default, potencial_dupla_barreira


def potencial_torch(x: torch.Tensor, V_bias: float, p: Dict) -> torch.Tensor:
    xb1, wb, ww = p["x_b1"], p["w_b"], p["w_w"]
    xb2 = xb1 + wb + ww
    V0 = p["V0"]
    V = torch.zeros_like(x)
    V = torch.where((x >= xb1) & (x < xb1 + wb), torch.full_like(x, V0), V)
    V = torch.where((x >= xb2) & (x < xb2 + wb), torch.full_like(x, V0), V)
    V = V - V_bias * (x / p["L_total"])
    return V


def residuos_schrodinger(
    rede: RedePINN_DBRTD,
    xE: torch.Tensor,
    V_bias: float = 0.0,
    p: Dict = None,
) -> torch.Tensor:
    if p is None:
        p = parametros_dbrtd_default()
    x = xE[:, 0:1]
    E = xE[:, 1:2]
    psi_R, psi_I = rede.psi(xE)

    def d2(u):
        du = torch.autograd.grad(u, xE, torch.ones_like(u), create_graph=True)[0][:, 0:1]
        d2u = torch.autograd.grad(du, xE, torch.ones_like(du), create_graph=True)[0][:, 0:1]
        return d2u

    d2R, d2I = d2(psi_R), d2(psi_I)
    V = potencial_torch(x, V_bias, p)
    h2m = p["hbar2_2m"]
    # −h2m ψ'' + V ψ − E ψ = 0
    R_R = -h2m * d2R + (V - E) * psi_R
    R_I = -h2m * d2I + (V - E) * psi_I
    return R_R, R_I


def perda_dbrtd(
    rede: RedePINN_DBRTD,
    xE_col: torch.Tensor,
    xE_L: torch.Tensor,
    xE_R: torch.Tensor,
    V_bias: float = 0.0,
    p: Dict = None,
    peso_pde: float = 1.0,
    peso_bc: float = 5.0,
) -> Tuple[torch.Tensor, dict]:
    if p is None:
        p = parametros_dbrtd_default()
    RR, RI = residuos_schrodinger(rede, xE_col, V_bias, p)
    perda_pde = torch.mean(RR ** 2) + torch.mean(RI ** 2)

    h2m = p["hbar2_2m"]
    # BC esquerda
    psiR, psiI = rede.psi(xE_L)
    gR = torch.autograd.grad(psiR, xE_L, torch.ones_like(psiR), create_graph=True)[0]
    gI = torch.autograd.grad(psiI, xE_L, torch.ones_like(psiI), create_graph=True)[0]
    dR, dI = gR[:, 0:1], gI[:, 0:1]
    E = xE_L[:, 1:2]
    kL = torch.sqrt(torch.clamp(E / h2m, min=1e-6))
    # ψ' − i k ψ − 2 i k  (A=1)
    # (dR + k ψI) + i(dI − k ψR) ≈ 0 + 2k i  → dR+k ψI≈0, dI−k ψR≈2k
    bc_L = torch.mean((dR + kL * psiI) ** 2) + torch.mean((dI - kL * psiR - 2 * kL) ** 2)

    # BC direita: ψ' + i k_R ψ = 0
    psiR, psiI = rede.psi(xE_R)
    gR = torch.autograd.grad(psiR, xE_R, torch.ones_like(psiR), create_graph=True)[0]
    gI = torch.autograd.grad(psiI, xE_R, torch.ones_like(psiI), create_graph=True)[0]
    dR, dI = gR[:, 0:1], gI[:, 0:1]
    E = xE_R[:, 1:2]
    kR = torch.sqrt(torch.clamp((E + V_bias) / h2m, min=1e-6))
    bc_R = torch.mean((dR - kR * psiI) ** 2) + torch.mean((dI + kR * psiR) ** 2)

    perda_bc = bc_L + bc_R
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, {"pde": float(perda_pde.detach()), "bc": float(perda_bc.detach())}
