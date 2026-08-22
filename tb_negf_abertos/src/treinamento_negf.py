"""Treinamento PINN TB-NEGF."""
import torch
import numpy as np
from typing import Dict, Optional
from .rede_pinn_negf import RedePINN_NEGF
from .residuo_negf import perda_negf


def treinar_negf(
    rede: RedePINN_NEGF,
    E_batch: torch.Tensor,
    H: np.ndarray,
    p: Dict,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_negf(rede, E_batch, H, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:5d} | perda={val:.4e} | dyson={det['dyson']:.4e} | causal={det['causal']:.4e}")
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
