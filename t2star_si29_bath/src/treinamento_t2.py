"""Treinamento PINN T₂*."""
import torch
from typing import Dict
from .rede_pinn_t2 import RedePINN_T2
from .residuo_t2 import perda_t2


def treinar_t2(
    rede: RedePINN_T2,
    t_col: torch.Tensor,
    T2s_fisico: float,
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
        total, det = perda_t2(rede, t_col, T2s_fisico)
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
                f"pde={det['pde']:.3e} | T2*_θ={det['T2s']:.4f} (fis={T2s_fisico:.4f})"
            )
        if epoca % 700 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
