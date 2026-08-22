"""Treinamento PINN Kadanoff–Baym."""
import torch
from typing import Dict, Optional
from .rede_pinn_kb import RedePINN_KB
from .residuo_kb import perda_kb
from .fisica_kb import parametros_kb_default


def treinar_kb(
    rede: RedePINN_KB,
    t12, t12_swap,
    p: Optional[Dict] = None,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    if p is None:
        p = parametros_kb_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_kb(rede, t12, t12_swap, p)
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
                f"kb={det['kb']:.4e} caus={det['caus']:.4e} herm={det['herm']:.4e}"
            )
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
