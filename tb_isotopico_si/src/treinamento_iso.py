"""Treinamento PINN isotópico."""
import torch
import numpy as np
from typing import Dict, Optional
from .rede_pinn_iso import RedePINN_Iso
from .residuo_iso import perda_iso


def treinar_iso(
    rede: RedePINN_Iso,
    H: np.ndarray,
    C_ref: Optional[np.ndarray] = None,
    n_epocas: int = 2000,
    taxa: float = 1e-3,
    verbose_cada: int = 200,
) -> Dict:
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_iso(rede, H, C_ref=C_ref)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:5d} | perda={val:.4e} | pde={det['pde']:.3e} | E={det['E']:.4f} | ||C||²={det['norm']:.3f}")
        if epoca % 700 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
