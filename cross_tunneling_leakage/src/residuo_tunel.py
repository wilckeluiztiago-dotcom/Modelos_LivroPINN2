"""
Resíduos das linhas de transmissão acopladas por tunelamento:

  ∂I1/∂z + C1 ∂V1/∂t = −J_leak
  ∂I2/∂z + C2 ∂V2/∂t = +J_leak
  ∂V1/∂z + L1 ∂I1/∂t + R1 I1 = 0
  ∂V2/∂z + L2 ∂I2/∂t + R2 I2 = 0

J_leak = G_eff (V1 − V2)
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_tunel import RedePINN_Tunel
from .fisica_tunel import parametros_tunel_default, G_tunel_wkb


def residuos_tunel(
    rede: RedePINN_Tunel,
    zt: torch.Tensor,
    p: Dict[str, float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_tunel_default()
    V1, V2, I1, I2 = rede.campos(zt)

    def grads(u):
        g = torch.autograd.grad(u, zt, torch.ones_like(u), create_graph=True)[0]
        return g[:, 0:1], g[:, 1:2]  # ∂z, ∂t

    dI1_dz, dI1_dt = grads(I1)
    dI2_dz, dI2_dt = grads(I2)
    dV1_dz, dV1_dt = grads(V1)
    dV2_dz, dV2_dt = grads(V2)

    G = p["G_leak0"] * G_tunel_wkb(p["d_int"], p["Phi_B"], p["m_star"])
    Jleak = G * (V1 - V2)

    R_cont1 = dI1_dz + p["C1"] * dV1_dt + Jleak
    R_cont2 = dI2_dz + p["C2"] * dV2_dt - Jleak
    R_tele1 = dV1_dz + p["L1"] * dI1_dt + p["R1"] * I1
    R_tele2 = dV2_dz + p["L2"] * dI2_dt + p["R2"] * I2
    return R_cont1, R_cont2, R_tele1, R_tele2


def perda_tunel(
    rede: RedePINN_Tunel,
    zt_col: torch.Tensor,
    zt_bc: torch.Tensor,
    V_bc: torch.Tensor,   # (N_bc, 2) = (V1, V2) nas bordas
    p: Dict[str, float] = None,
    peso_pde: float = 1.0,
    peso_bc: float = 8.0,
) -> Tuple[torch.Tensor, dict]:
    R1, R2, R3, R4 = residuos_tunel(rede, zt_col, p)
    perda_pde = (
        torch.mean(R1 ** 2) + torch.mean(R2 ** 2)
        + torch.mean(R3 ** 2) + torch.mean(R4 ** 2)
    )
    V1b, V2b, _, _ = rede.campos(zt_bc)
    perda_bc = torch.mean((V1b - V_bc[:, 0:1]) ** 2) + torch.mean((V2b - V_bc[:, 1:2]) ** 2)
    total = peso_pde * perda_pde + peso_bc * perda_bc
    return total, {
        "pde": float(perda_pde.detach()),
        "bc": float(perda_bc.detach()),
    }
