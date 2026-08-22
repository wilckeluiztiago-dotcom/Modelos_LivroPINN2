"""
Resíduos:
  1) G_θ(V,T) ≈ G_2CK analítico
  2) ∂t ρ + i[H,ρ] − K_mem[ρ] ≈ 0  (Markov + memória exponencial simplificada)
  3) Tr ρ = 1 (já embutido)
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_2ck import RedePINN_G, RedePINN_Rho
from .fisica_2ck import parametros_2ck_default


def perda_G(
    rede_G: RedePINN_G,
    VT: torch.Tensor,
    G_ref: torch.Tensor,
) -> Tuple[torch.Tensor, dict]:
    G = rede_G(VT)
    perda = torch.mean((G - G_ref) ** 2)
    return perda, {"G": float(perda.detach())}


def perda_rho(
    rede_rho: RedePINN_Rho,
    t: torch.Tensor,
    p: Dict = None,
    Omega: float = 0.3,
    gamma: float = 0.5,
    tau_mem: float = 0.8,
) -> Tuple[torch.Tensor, dict]:
    """
    Dinâmica efetiva:
      ∂t ρ = −i[H,ρ] − γ (ρ − ρ_eq) − (1/τ) ∫ e^{−s/τ} (ρ(t)−ρ(t−s)) ds
    Aproximamos memória por: −(ρ − ρ_eq)/τ_mem extra
    H = (Ω/2) σ_z  (campo efetivo / desdobramento)
    """
    if p is None:
        p = parametros_2ck_default()
    a, c, d, b = rede_rho.matriz_rho(t)

    def dt(u):
        return torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]

    da, dc, dd = dt(a), dt(c), dt(d)

    # [H,ρ] com H = (Ω/2) σ_z → acopla as coerências
    # ρ01_dot da comutação: −i Ω ρ01  →  dc e dd
    # equações de Bloch-like
    R_a = da + gamma * (a - 0.5)           # população → 1/2
    R_c = dc + Omega * d + gamma * c + c / tau_mem
    R_d = dd - Omega * c + gamma * d + d / tau_mem

    perda = torch.mean(R_a ** 2) + torch.mean(R_c ** 2) + torch.mean(R_d ** 2)
    # pureza / positividade suave
    pureza = a * b - (c ** 2 + d ** 2)
    perda_pos = torch.mean(torch.relu(-pureza) ** 2)
    total = perda + 0.5 * perda_pos
    return total, {
        "rho": float(perda.detach()),
        "pos": float(perda_pos.detach()),
    }
