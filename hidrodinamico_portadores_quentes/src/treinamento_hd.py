"""Treinamento PINN hidrodinâmico."""
import torch
from typing import Dict, Optional
from .rede_pinn_hd import RedePINN_HD
from .residuo_hd import perda_hd
from .fisica_hd import parametros_hd_default


def treinar_hd(
    rede: RedePINN_HD,
    x_col, x_bc, n_bc, Tn_bc,
    p: Optional[Dict] = None,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    if p is None:
        p = parametros_hd_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_hd(rede, x_col, x_bc, n_bc, Tn_bc, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:5d} | perda={val:.4e} | pde={det['pde']:.4e} | bc={det['bc']:.4e}")
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
