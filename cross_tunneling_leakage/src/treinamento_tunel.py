"""Treinamento PINN cross-tunneling."""
import torch
from typing import Dict, Optional
from .rede_pinn_tunel import RedePINN_Tunel
from .residuo_tunel import perda_tunel
from .fisica_tunel import parametros_tunel_default


def treinar_tunel(
    rede: RedePINN_Tunel,
    zt_col: torch.Tensor,
    zt_bc: torch.Tensor,
    V_bc: torch.Tensor,
    p: Optional[Dict] = None,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    if p is None:
        p = parametros_tunel_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None

    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_tunel(rede, zt_col, zt_bc, V_bc, p)
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
