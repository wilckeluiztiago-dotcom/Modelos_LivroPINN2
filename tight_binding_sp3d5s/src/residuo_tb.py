"""
Resíduo nos sítios atômicos:

  Σ_{jβ} H_{iα,jβ} C_{jβ} + V_P C_{iα} ≈ E C_{iα}

C_α(R_i) vem da rede contínua.
+ normalização Σ_i Σ_α |C|² ≈ 1
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .rede_pinn_tb import RedePINN_TB
from .fisica_tb import N_ORB


def perda_tb_sitios(
    rede: RedePINN_TB,
    H: np.ndarray,
    pos: np.ndarray,
    peso_pde: float = 1.0,
    peso_norm: float = 5.0,
    peso_match: float = 2.0,
    C_ref: np.ndarray = None,  # autovetor TB de referência (opcional)
) -> Tuple[torch.Tensor, dict]:
    device = next(rede.parameters()).device
    n_at = len(pos)
    dim = n_at * N_ORB

    r = torch.tensor(pos, dtype=torch.float32, device=device)
    C_sites = rede(r)  # (n_at, 10)
    C_vec = C_sites.reshape(-1)  # (dim,)

    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    E = rede.energia()
    residual = H_t @ C_vec - E * C_vec
    perda_pde = torch.mean(residual ** 2)

    norma = torch.sum(C_vec ** 2)
    perda_norm = (norma - 1.0) ** 2

    perda_match = torch.tensor(0.0, device=device)
    if C_ref is not None:
        cref = torch.tensor(C_ref, dtype=torch.float32, device=device)
        # alinhar sinal
        if torch.dot(C_vec, cref) < 0:
            cref = -cref
        perda_match = torch.mean((C_vec - cref) ** 2)

    total = peso_pde * perda_pde + peso_norm * perda_norm + peso_match * perda_match
    return total, {
        "pde": float(perda_pde.detach()),
        "norm": float(perda_norm.detach()),
        "match": float(perda_match.detach()),
        "E": float(E.detach()),
        "norma": float(norma.detach()),
    }
