"""
Resíduo: (E I - H - Σ_L - Σ_R) G^R − I = 0
"""

import torch
import numpy as np
from typing import Dict, Tuple
from .rede_pinn_negf import RedePINN_NEGF
from .fisica_negf import Sigma_contatos


def perda_negf(
    rede: RedePINN_NEGF,
    E_batch: torch.Tensor,
    H: np.ndarray,
    p: Dict,
    peso_dyson: float = 1.0,
    peso_herm: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    device = E_batch.device
    n = H.shape[0]
    H_t = torch.tensor(H.real, dtype=torch.float32, device=device)  # H hermitiano real aqui
    I = torch.eye(n, device=device)

    ReG, ImG = rede(E_batch)
    # residual complexo por ponto de energia
    perda_d = torch.tensor(0.0, device=device)
    for b in range(E_batch.shape[0]):
        E = float(E_batch[b, 0].detach())
        SL, SR = Sigma_contatos(E, n, p)
        # Σ total
        S_re = torch.tensor((SL + SR).real, dtype=torch.float32, device=device)
        S_im = torch.tensor((SL + SR).imag, dtype=torch.float32, device=device)
        eta = p["eta"]
        # A = (E I - H - ReΣ) + i(η I - ImΣ)
        Ar = E * I - H_t - S_re
        Ai = eta * I - S_im
        Gr, Gi = ReG[b], ImG[b]
        # (Ar + i Ai)(Gr + i Gi) = I + 0i
        # Ar Gr - Ai Gi = I
        # Ar Gi + Ai Gr = 0
        R1 = Ar @ Gr - Ai @ Gi - I
        R2 = Ar @ Gi + Ai @ Gr
        perda_d = perda_d + torch.mean(R1 ** 2) + torch.mean(R2 ** 2)
    perda_d = perda_d / E_batch.shape[0]

    # suave: G deve ser aproximadamente "causal" Im diag ≤ 0
    perda_c = torch.mean(torch.relu(torch.diagonal(ImG, dim1=1, dim2=2)) ** 2)

    total = peso_dyson * perda_d + peso_herm * perda_c
    return total, {
        "dyson": float(perda_d.detach()),
        "causal": float(perda_c.detach()),
    }
