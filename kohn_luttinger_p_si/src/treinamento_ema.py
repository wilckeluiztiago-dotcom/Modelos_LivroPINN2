"""Treinamento EMA / valley-orbit PINN."""
import torch
from typing import Dict, Optional
from .rede_pinn_ema import BancoEMA
from .residuo_ema import perda_ema
from .fisica_ema import parametros_ema_default


def treinar_ema(
    banco: BancoEMA,
    r_col, r_quad,
    p: Optional[Dict] = None,
    n_epocas: int = 3000,
    taxa: float = 1e-3,
    verbose_cada: int = 300,
) -> Dict:
    if p is None:
        p = parametros_ema_default()
    opt = torch.optim.Adam(banco.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_ema(banco, r_col, r_quad, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in banco.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            Ed = banco.energias_dict()
            print(
                f"  época {epoca:5d} | perda={val:.4e} | "
                f"pde={det['pde']:.3e} norm={det['norm']:.3e} | "
                f"E={{{', '.join(f'{k}:{v:.3f}' for k,v in Ed.items())}}}"
            )
        if epoca % 1000 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        banco.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
