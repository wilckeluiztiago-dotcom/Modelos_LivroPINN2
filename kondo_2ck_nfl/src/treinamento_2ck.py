"""Treinamento PINN 2CK."""
import torch
from typing import Dict, Optional
from .rede_pinn_2ck import RedePINN_G, RedePINN_Rho
from .residuo_2ck import perda_G, perda_rho
from .fisica_2ck import parametros_2ck_default


def treinar_G(
    rede: RedePINN_G,
    VT, G_ref,
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
        total, det = perda_G(rede, VT, G_ref)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  [G] época {epoca:5d} | perda={val:.4e}")
        if epoca % 700 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}


def treinar_rho(
    rede: RedePINN_Rho,
    t,
    p: Optional[Dict] = None,
    n_epocas: int = 2000,
    taxa: float = 1e-3,
    verbose_cada: int = 200,
) -> Dict:
    if p is None:
        p = parametros_2ck_default()
    opt = torch.optim.Adam(rede.parameters(), lr=taxa)
    historico = []
    melhor = float("inf")
    best = None
    for epoca in range(1, n_epocas + 1):
        opt.zero_grad()
        total, det = perda_rho(rede, t, p)
        total.backward()
        opt.step()
        val = float(total.detach())
        historico.append(val)
        if val < melhor:
            melhor = val
            best = {k: v.detach().cpu().clone() for k, v in rede.state_dict().items()}
        if verbose_cada and epoca % verbose_cada == 0:
            print(f"  [ρ] época {epoca:5d} | perda={val:.4e} | pos={det['pos']:.3e}")
        if epoca % 700 == 0:
            for g in opt.param_groups:
                g["lr"] *= 0.7
    if best is not None:
        rede.load_state_dict(best)
    return {"historico": historico, "perda_final": melhor}
