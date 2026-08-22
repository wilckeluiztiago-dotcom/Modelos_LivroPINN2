"""Treinamento PINN eletromigração Korhonen."""
import torch
from typing import Dict, Optional
from .rede_pinn_em import RedePotencial, RedeTensao
from .residuo_em import perda_em
from .fisica_korhonen import parametros_korhonen_default


def treinar_em(
    rede_sigma: RedeTensao,
    rede_phi: RedePotencial,
    x_phi, xt_col, x_bc, t_bc, phi_bc_val, sigma0_xt, sigma0_val,
    p: Optional[Dict] = None,
    n_epocas: int = 2500,
    taxa: float = 1e-3,
    verbose_cada: int = 250,
) -> Dict:
    if p is None:
        p = parametros_korhonen_default()
    params = list(rede_sigma.parameters()) + list(rede_phi.parameters())
    opt = torch.optim.Adam(params, lr=taxa)
    historico = []
    melhor = float("inf")
    best_s, best_p = None, None

    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_em(
            rede_sigma, rede_phi,
            x_phi, xt_col, x_bc, t_bc, phi_bc_val, sigma0_xt, sigma0_val, p,
        )
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best_s = {k: v.detach().cpu().clone() for k, v in rede_sigma.state_dict().items()}
            best_p = {k: v.detach().cpu().clone() for k, v in rede_phi.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(
                f"  época {epoca:5d} | perda={val:.4e} | "
                f"pde={det['pde']:.3e} flux={det['flux']:.3e} "
                f"bc={det['bc']:.3e} ic={det['ic']:.3e}"
            )
        if epoca % 800 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7

    if best_s is not None:
        rede_sigma.load_state_dict(best_s)
        rede_phi.load_state_dict(best_p)
    return {"historico": historico, "perda_final": melhor}
