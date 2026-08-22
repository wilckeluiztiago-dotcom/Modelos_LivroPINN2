"""Treinamento PINN termoelétrico com PyTorch."""
import torch
from typing import Dict, Optional
from .rede_pinn_termo import RedePINN_Termo
from .residuo_termo import perda_termo
from .fisica_termo import parametros_termo_default


def treinar_termo(
    rede: RedePINN_Termo,
    x_col: torch.Tensor,
    x_bc: torch.Tensor,
    phi_bc: torch.Tensor,
    T_bc: torch.Tensor,
    p: Optional[Dict] = None,
    n_epocas: int = 2000,
    taxa: float = 1e-3,
    verbose_cada: int = 200,
) -> Dict:
    if p is None:
        p = parametros_termo_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    melhor_state = None

    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, pde, bc = perda_termo(rede, x_col, x_bc, phi_bc, T_bc, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            melhor_state = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  época {epoca:5d} | perda={val:.4e} | pde={float(pde):.4e} | bc={float(bc):.4e}")
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7

    if melhor_state is not None:
        rede.load_state_dict(melhor_state)
    return {"historico": historico, "perda_final": melhor}
