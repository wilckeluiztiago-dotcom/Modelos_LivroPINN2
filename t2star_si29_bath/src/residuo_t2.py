"""
Resíduos:

1) FID gaussiano:  ∂t S_x + 2 t / (T₂*)²  S_x  ≈ 0
   (derivada de exp(−(t/T2*)²))

2) T₂* físico: T₂*_θ ≈ √2 / √(Σ A_k²)

3) IC: S_x(0) = 1
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_t2 import RedePINN_T2


def residuo_fid(
    rede: RedePINN_T2,
    t: torch.Tensor,
) -> torch.Tensor:
    S = rede(t)
    dS = torch.autograd.grad(S, t, torch.ones_like(S), create_graph=True)[0]
    T2s = rede.T2_star()
    # d/dt exp(-(t/T)^2) = -2t/T² exp(...)
    return dS + (2.0 * t / (T2s ** 2 + 1e-12)) * S


def perda_t2(
    rede: RedePINN_T2,
    t_col: torch.Tensor,
    T2s_fisico: float,
    peso_pde: float = 1.0,
    peso_T2: float = 5.0,
    peso_ic: float = 10.0,
) -> Tuple[torch.Tensor, dict]:
    R = residuo_fid(rede, t_col)
    perda_pde = torch.mean(R ** 2)

    T2s = rede.T2_star()
    perda_T2 = (T2s - T2s_fisico) ** 2

    t0 = torch.zeros(1, 1, device=t_col.device)
    S0 = rede(t0)
    perda_ic = (S0 - 1.0) ** 2

    total = peso_pde * perda_pde + peso_T2 * perda_T2 + peso_ic * perda_ic.squeeze()
    return total, {
        "pde": float(perda_pde.detach()),
        "T2": float(perda_T2.detach()),
        "ic": float(perda_ic.detach().squeeze()),
        "T2s": float(T2s.detach()),
    }
