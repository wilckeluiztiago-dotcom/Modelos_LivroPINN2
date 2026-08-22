"""Treinamento dos modos transversais."""
import torch
from typing import Dict
from .rede_pinn_modos import BancoModos
from .residuo_modos import perda_modos


def treinar_modos(
    banco: BancoModos,
    y_col: torch.Tensor,
    y_quad: torch.Tensor,
    W: float = 1.0,
    n_epocas: int = 3000,
    taxa: float = 1e-3,
    verbose_cada: int = 300,
) -> Dict:
    opt = torch.optim.Adam(banco.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None

    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_modos(banco, y_col, y_quad, W)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in banco.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            E = banco.energias().detach().cpu().numpy()
            print(
                f"  época {epoca:5d} | perda={val:.4e} | "
                f"pde={det['pde']:.3e} ortho={det['ortho']:.3e} bc={det['bc']:.3e} | "
                f"E={E.round(3)}"
            )
        if epoca % 1000 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7

    if best is not None:
        banco.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
