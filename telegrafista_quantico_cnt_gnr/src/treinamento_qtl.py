"""Treinamento PINN telegrafista quântico."""
import torch
from typing import Dict, Optional
from .rede_pinn_qtl import RedePINN_QTL
from .residuo_qtl import perda_qtl
from .fisica_telegrafista import parametros_qtl_default


def treinar_qtl(
    rede: RedePINN_QTL,
    zt_col, zt_src, V_src, zt_load,
    p: Optional[Dict] = None,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    if p is None:
        p = parametros_qtl_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None

    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_qtl(rede, zt_col, zt_src, V_src, zt_load, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(
                f"  época {epoca:5d} | perda={val:.4e} | "
                f"pde={det['pde']:.4e} src={det['src']:.4e} term={det['term']:.4e}"
            )
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7

    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
