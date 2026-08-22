"""
Resíduo PINN:

  H ψ_n = E_n ψ_n
  H = −(ħ²/2m) ∂_{yy} + V_conf(y)

  + ortogonalidade ∫ ψ_n ψ_m dy = δ_{nm}
  + BC ψ(0)=ψ(W)=0
  + continuidade de probabilidade (estacionária): ∂_y j_y ≈ 0
"""

import torch
from typing import Tuple
from .rede_pinn_modos import BancoModos


def pot_conf_torch(y: torch.Tensor, W: float = 1.0, V_wall: float = 50.0) -> torch.Tensor:
    V = torch.zeros_like(y)
    V = torch.where(y < 0, torch.full_like(y, V_wall), V)
    V = torch.where(y > W, torch.full_like(y, V_wall), V)
    return V


def residuo_schrodinger(
    banco: BancoModos,
    y: torch.Tensor,
    W: float = 1.0,
    hbar: float = 1.0,
    m_star: float = 1.0,
) -> torch.Tensor:
    E = banco.energias()
    residuos = []
    for n in range(banco.n_modos):
        psi = banco.psi(n, y)
        dpsi = torch.autograd.grad(psi, y, torch.ones_like(psi), create_graph=True)[0]
        d2psi = torch.autograd.grad(dpsi, y, torch.ones_like(dpsi), create_graph=True)[0]
        V = pot_conf_torch(y, W)
        Hpsi = -(hbar ** 2 / (2 * m_star)) * d2psi + V * psi
        residuos.append(Hpsi - E[n] * psi)
    return torch.cat(residuos, dim=0)


def perda_ortogonalidade(
    banco: BancoModos,
    y_quad: torch.Tensor,
    W: float = 1.0,
) -> torch.Tensor:
    """∫ ψ_n ψ_m dy ≈ δ_{nm} via regra do trapézio."""
    dy = (y_quad[1] - y_quad[0]).detach()
    psis = [banco.psi(n, y_quad).squeeze() for n in range(banco.n_modos)]
    perda = torch.tensor(0.0, device=y_quad.device)
    for n in range(banco.n_modos):
        for m in range(n, banco.n_modos):
            integ = torch.sum(psis[n] * psis[m]) * dy
            alvo = 1.0 if n == m else 0.0
            perda = perda + (integ - alvo) ** 2
    return perda


def perda_bc(
    banco: BancoModos,
    W: float = 1.0,
) -> torch.Tensor:
    device = next(banco.parameters()).device
    y0 = torch.tensor([[0.0]], device=device)
    yW = torch.tensor([[W]], device=device)
    perda = torch.tensor(0.0, device=device)
    for n in range(banco.n_modos):
        perda = perda + banco.psi(n, y0).pow(2).mean() + banco.psi(n, yW).pow(2).mean()
    return perda


def perda_modos(
    banco: BancoModos,
    y_col: torch.Tensor,
    y_quad: torch.Tensor,
    W: float = 1.0,
    peso_pde: float = 1.0,
    peso_ortho: float = 5.0,
    peso_bc: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    R = residuo_schrodinger(banco, y_col, W)
    perda_pde = torch.mean(R ** 2)
    perda_orth = perda_ortogonalidade(banco, y_quad, W)
    perda_b = perda_bc(banco, W)
    total = peso_pde * perda_pde + peso_ortho * perda_orth + peso_bc * perda_b
    return total, {
        "pde": float(perda_pde.detach()),
        "ortho": float(perda_orth.detach()),
        "bc": float(perda_b.detach()),
    }
