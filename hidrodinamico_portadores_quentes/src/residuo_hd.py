"""
Resíduos estacionários 1D (∂t → 0):

  d/dx (n v) = 0

  d/dx (n v²) + (q/m*) n E + (1/m*) d/dx (n kB Tn) + n v / τ_p = 0

  d/dx (v n W) + d/dx (−κ dTn/dx) + q n v E? wait:
  energy: ∇·(v n W) = −q n v · E − ∇·Q − n (W−W0)/τ_w
  with Q = −κ ∇T
  and E = −dφ/dx → −q n v · E = q n v dφ/dx
  Here E_field is the electric field magnitude in +x driving force:
  force on electron is −q E_vec with E_vec = E_field * xhat in our sign convention
  from momentum: −(q/m) E in the equation as written by user with E = field.

User momentum:
  ∂t(nv) + ∇·(n v⊗v) = −(q n / m*) E − (1/m*) ∇(n kT) − n v / τ_p

Stationary 1D:
  d(n v)/dx = 0
  d(n v²)/dx + (q/m) n E + (1/m) d(n kT)/dx + n v/τ_p = 0

Energy stationary:
  d(v n W)/dx + d(−κ dT/dx)/dx + q n v E + n (W−W0)/τ_w = 0
  (sign of Joule: electrons heated by field work)
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_hd import RedePINN_HD
from .fisica_hd import parametros_hd_default


def residuos_hd(
    rede: RedePINN_HD,
    x: torch.Tensor,
    p: Dict = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_hd_default()
    n, v, Tn = rede.campos(x)

    def dx(u):
        return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

    dn, dv, dTn = dx(n), dx(v), dx(Tn)
    # campo com pico
    L = p["L"]
    E0 = p["E_field"]
    E = E0 * (1.0 + 0.5 * torch.exp(-((x - 0.5 * L) / 0.15) ** 2))

    # continuidade
    R_cont = dx(n * v)

    # momento
    tau_p = p["tau_p0"] * torch.sqrt(p["T0"] / torch.clamp(Tn, min=0.1))
    R_mom = (
        dx(n * v * v)
        + (p["q"] / p["m_star"]) * n * E
        + (1.0 / p["m_star"]) * dx(n * p["kB"] * Tn)
        + n * v / tau_p
    )

    # energia
    W = 1.5 * p["kB"] * Tn + 0.5 * p["m_star"] * v * v
    W0 = 1.5 * p["kB"] * p["T0"]
    tau_w = p["tau_w0"] * (p["T0"] / torch.clamp(Tn, min=0.1))
    Q = -p["kappa_n"] * dTn
    R_en = (
        dx(v * n * W)
        + dx(Q)
        + p["q"] * n * v * E
        + n * (W - W0) / tau_w
    )
    return R_cont, R_mom, R_en


def perda_hd(
    rede: RedePINN_HD,
    x_col: torch.Tensor,
    x_bc: torch.Tensor,
    n_bc: torch.Tensor,
    Tn_bc: torch.Tensor,
    p: Dict = None,
    peso_pde: float = 1.0,
    peso_bc: float = 8.0,
) -> Tuple[torch.Tensor, dict]:
    Rc, Rm, Re = residuos_hd(rede, x_col, p)
    perda_pde = torch.mean(Rc ** 2) + torch.mean(Rm ** 2) + torch.mean(Re ** 2)
    n_b, v_b, Tn_b = rede.campos(x_bc)
    perda_bc = torch.mean((n_b - n_bc) ** 2) + torch.mean((Tn_b - Tn_bc) ** 2)
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, {
        "pde": float(perda_pde.detach()),
        "bc": float(perda_bc.detach()),
    }
