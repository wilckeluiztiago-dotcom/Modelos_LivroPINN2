"""
Resíduo: (H0 + diag(δϵ) + V_P) C = E C
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional
from .rede_pinn_iso import RedePINN_Iso


def perda_iso(
    rede: RedePINN_Iso,
    H: np.ndarray,
    peso_pde: float = 1.0,
    peso_norm: float = 5.0,
    C_ref: Optional[np.ndarray] = None,
    peso_match: float = 1.0,
) -> Tuple[torch.Tensor, dict]:
    device = next(rede.parameters()).device
    n = H.shape[0]
    H_t = torch.tensor(H, dtype=torch.float32, device=device)
    C = rede.vetor_C(device)
    E = rede.energia()
    res = H_t @ C - E * C
    perda_pde = torch.mean(res ** 2)
    norma = torch.sum(C ** 2)
    perda_norm = (norma - 1.0) ** 2
    perda_m = torch.tensor(0.0, device=device)
    if C_ref is not None:
        cref = torch.tensor(C_ref, dtype=torch.float32, device=device)
        if torch.dot(C, cref) < 0:
            cref = -cref
        perda_m = torch.mean((C - cref) ** 2)
    total = peso_pde * perda_pde + peso_norm * perda_norm + peso_match * perda_m
    return total, {
        "pde": float(perda_pde.detach()),
        "norm": float(norma.detach()),
        "E": float(E.detach()),
    }
