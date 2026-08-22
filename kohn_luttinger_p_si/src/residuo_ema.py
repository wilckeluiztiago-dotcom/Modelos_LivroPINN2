"""
Resíduo radial EMA (1D efetivo):

  −(ħ²/2m*) (F'' + 2/r F') + V(r) F = E F

com V = V_coul + V_cc.
+ normalização ∫ |F|² 4π r² dr = 1
+ ortogonalidade entre canais
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_ema import BancoEMA
from .fisica_ema import parametros_ema_default


def V_torch(r: torch.Tensor, p: Dict) -> torch.Tensor:
    Vc = -1.0 / (p["eps_r"] * torch.clamp(r, min=1e-5))
    Vcc = -p["V0"] * torch.exp(-r / p["r0"])
    return Vc + Vcc


def residuo_radial(
    banco: BancoEMA,
    r: torch.Tensor,
    p: Dict = None,
) -> torch.Tensor:
    if p is None:
        p = parametros_ema_default()
    m = p["m_star"]
    res_list = []
    for s in banco.SIMETRIAS:
        F = banco.F(s, r)
        E = banco.energia(s)
        dF = torch.autograd.grad(F, r, torch.ones_like(F), create_graph=True)[0]
        d2F = torch.autograd.grad(dF, r, torch.ones_like(dF), create_graph=True)[0]
        # laplaciano radial de F: F'' + 2/r F'
        lap = d2F + 2.0 / torch.clamp(r, min=1e-5) * dF
        H_F = -0.5 / m * lap + V_torch(r, p) * F
        res_list.append(H_F - E * F)
    return torch.cat(res_list, dim=0)


def perda_norm_ortho(
    banco: BancoEMA,
    r_quad: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """∫ F² 4π r² dr ≈ 1; ortogonalidade entre canais."""
    dr = (r_quad[1] - r_quad[0]).detach()
    Fs = {s: banco.F(s, r_quad).squeeze() for s in banco.SIMETRIAS}
    peso = 4.0 * 3.14159265 * r_quad.squeeze() ** 2
    perda_n = torch.tensor(0.0, device=r_quad.device)
    perda_o = torch.tensor(0.0, device=r_quad.device)
    sims = banco.SIMETRIAS
    for i, s in enumerate(sims):
        integ = torch.sum(Fs[s] ** 2 * peso) * dr
        perda_n = perda_n + (integ - 1.0) ** 2
        for s2 in sims[i + 1:]:
            cross = torch.sum(Fs[s] * Fs[s2] * peso) * dr
            perda_o = perda_o + cross ** 2
    return perda_n, perda_o


def perda_ema(
    banco: BancoEMA,
    r_col: torch.Tensor,
    r_quad: torch.Tensor,
    p: Dict = None,
    peso_pde: float = 1.0,
    peso_norm: float = 5.0,
    peso_ortho: float = 2.0,
    peso_split: float = 3.0,
) -> Tuple[torch.Tensor, dict]:
    """
    peso_split: empurra E_A1 < E_T2 < E_E (mais ligados = mais negativos)
    alinhado à ordem experimental.
    """
    if p is None:
        p = parametros_ema_default()
    R = residuo_radial(banco, r_col, p)
    perda_pde = torch.mean(R ** 2)
    pn, po = perda_norm_ortho(banco, r_quad)

    EA1 = banco.energia("A1")
    ET2 = banco.energia("T2")
    EE = banco.energia("E")
    # A1 mais ligado (menor E), depois T2, depois E
    split = torch.relu(EA1 - ET2) ** 2 + torch.relu(ET2 - EE) ** 2

    total = (
        peso_pde * perda_pde
        + peso_norm * pn
        + peso_ortho * po
        + peso_split * split
    )
    return total, {
        "pde": float(perda_pde.detach()),
        "norm": float(pn.detach()),
        "ortho": float(po.detach()),
        "split": float(split.detach()),
    }
