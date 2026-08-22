"""Treinamento PINN DBRTD."""
import torch
from typing import Dict, Optional
from .rede_pinn_dbrtd import RedePINN_DBRTD
from .residuo_dbrtd import perda_dbrtd
from .fisica_dbrtd import parametros_dbrtd_default


def treinar_dbrtd(
    rede: RedePINN_DBRTD,
    xE_col, xE_L, xE_R,
    V_bias: float = 0.0,
    p: Optional[Dict] = None,
    n_epocas: int = 2000,
    taxa: float = 1e-3,
    verbose_cada: int = 200,
) -> Dict:
    if p is None:
        p = parametros_dbrtd_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_dbrtd(rede, xE_col, xE_L, xE_R, V_bias, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:5d} | perda={val:.4e} | pde={det['pde']:.4e} | bc={det['bc']:.4e}")
        if epoca % 700 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
