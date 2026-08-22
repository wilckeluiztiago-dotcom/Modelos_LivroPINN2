"""
Resíduo Kadanoff–Baym (orbital único, memória local):

  i ℏ ∂_{t1} G^< = (ε + Σ_HF) G^< + ∫ Σ^R G^< + Σ^< G^A

Aproximação de memória Markov+exponencial:
  ∫_0^{t1} K(t1−s) G^<(s,t2) ds  ≈  (γ) * média local

Também:
  G^R(t1,t2) ≈ 0 se t1 < t2  (causalidade)
  G^<(t1,t2) = −[G^<(t2,t1)]*  (hermiticidade anti)
"""

import torch
from typing import Dict, Tuple
from .rede_pinn_kb import RedePINN_KB
from .fisica_kb import parametros_kb_default


def residuos_kb(
    rede: RedePINN_KB,
    t12: torch.Tensor,
    p: Dict = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if p is None:
        p = parametros_kb_default()
    hbar = p["hbar"]
    eps = p["eps0"] + p["U_HF"] * p["n_eq"]
    gamma = p["gamma"]

    Gl_R, Gl_I = rede.G_lesser(t12)
    Gr_R, Gr_I = rede.G_retarded(t12)
    t1 = t12[:, 0:1]
    t2 = t12[:, 1:2]

    def dt1(u):
        return torch.autograd.grad(u, t12, torch.ones_like(u), create_graph=True)[0][:, 0:1]

    dGlR = dt1(Gl_R)
    dGlI = dt1(Gl_I)

    # i ∂t1 (Gl_R + i Gl_I) = (ε − i γ/2) (Gl_R + i Gl_I)  [Markov lead]
    # i (dR + i dI) = (ε − iγ/2)(R + i I)
    # i dR − dI = ε R − (γ/2) I + i[ε I + (γ/2) R]
    # Real: −dI = ε R − (γ/2) I
    # Imag:  dR = ε I + (γ/2) R
    # → residual:
    R1 = -dGlI - eps * Gl_R + 0.5 * gamma * Gl_I
    R2 = dGlR - eps * Gl_I - 0.5 * gamma * Gl_R
    # escala iℏ
    R1 = hbar * R1
    R2 = hbar * R2
    return R1, R2


def perda_kb(
    rede: RedePINN_KB,
    t12: torch.Tensor,
    t12_swap: torch.Tensor,
    p: Dict = None,
    peso_kb: float = 1.0,
    peso_caus: float = 2.0,
    peso_herm: float = 2.0,
) -> Tuple[torch.Tensor, dict]:
    if p is None:
        p = parametros_kb_default()
    R1, R2 = residuos_kb(rede, t12, p)
    perda_kb_ = torch.mean(R1 ** 2) + torch.mean(R2 ** 2)

    # causalidade G^R: se t1 < t2, G^R ≈ 0
    t1, t2 = t12[:, 0:1], t12[:, 1:2]
    Gr_R, Gr_I = rede.G_retarded(t12)
    mask = (t1 < t2).float()
    perda_caus = torch.mean(mask * (Gr_R ** 2 + Gr_I ** 2))

    # hermiticidade: G^<(t1,t2) + conj G^<(t2,t1) ≈ 0
    # i.e. Gl_R(t1,t2) + Gl_R(t2,t1) ≈ 0, Gl_I(t1,t2) − Gl_I(t2,t1) ≈ 0
    Gl_R, Gl_I = rede.G_lesser(t12)
    Gl_R_s, Gl_I_s = rede.G_lesser(t12_swap)
    perda_herm = torch.mean((Gl_R + Gl_R_s) ** 2) + torch.mean((Gl_I - Gl_I_s) ** 2)

    total = peso_kb * perda_kb_ + peso_caus * perda_caus + peso_herm * perda_herm
    return total, {
        "kb": float(perda_kb_.detach()),
        "caus": float(perda_caus.detach()),
        "herm": float(perda_herm.detach()),
    }
