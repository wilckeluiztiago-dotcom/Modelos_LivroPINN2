#!/usr/bin/env python3
"""
Paschen Modificado · Microplasma Nanométrico · Fowler–Nordheim · PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_paschen import (
    parametros_fn_default, tensao_paschen_classica, tensao_fn_gap, G_FN,
)
from src.rede_pinn_fn import RedePINN_FN
from src.treinamento_fn import treinar_fn


def principal():
    print("=" * 70)
    print("  Paschen / Microplasma FN · Gaps sub-5 nm · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_fn_default()
    print(f"\n[1] Device: {dev}")
    print(f"    A_FN={p['A_FN']}, B_FN={p['B_FN']}, d_gap={p['d_gap']}")

    # colocation
    n_col = 400
    x = torch.rand(n_col, 1, device=dev)
    t = torch.rand(n_col, 1, device=dev)
    xt_col = torch.cat([x, t], dim=1).requires_grad_(True)

    # BC: φ(0)=0 (cátodo), φ(1)=V_bias (ânodo)
    V_bias = 1.5
    n_bc = 40
    t_bc = torch.linspace(0, 1, n_bc, device=dev).reshape(-1, 1)
    x0 = torch.zeros(n_bc, 1, device=dev)
    x1 = torch.ones(n_bc, 1, device=dev)
    xt_bc = torch.cat([torch.cat([x0, t_bc], 1), torch.cat([x1, t_bc], 1)], 0)
    phi_bc = torch.cat([torch.zeros(n_bc, 1, device=dev), torch.full((n_bc, 1), V_bias, device=dev)], 0)

    print("\n[2] Treinando PINN (φ, n_e)...")
    rede = RedePINN_FN([2, 48, 48, 48, 2]).to(dev)
    print(f"    Parâmetros: {sum(q.numel() for q in rede.parameters())}")
    res = treinar_fn(rede, xt_col, xt_bc, phi_bc, p, n_epocas=2500, taxa=1e-3, verbose_cada=250)
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfil
    xg = torch.linspace(0, 1, 120, device=dev).reshape(-1, 1)
    xtg = torch.cat([xg, torch.full_like(xg, 0.5)], 1)
    with torch.no_grad():
        phi, ne = rede.campos(xtg)
    x_np = xg.cpu().numpy().ravel()
    phi_np = phi.cpu().numpy().ravel()
    ne_np = ne.cpu().numpy().ravel()
    E = -np.gradient(phi_np, x_np)

    # curvas Paschen vs FN
    d = np.linspace(0.3, 8, 80)
    V_paschen = tensao_paschen_classica(d * 1.0)  # pd ~ d
    V_fn = tensao_fn_gap(d, p["B_FN"])

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x_np, phi_np, "C0-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi$")
    ax.set_title(r"(a) Potencial no gap (cátodo→ânodo)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(x_np, ne_np, "C3-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$n_e$")
    ax.set_title(r"(b) Densidade eletrônica (emissão FN no cátodo)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(d, V_paschen, "k--", lw=1.5, label="Paschen clássico")
    ax.plot(d, V_fn, "C3-", lw=2, label="regime FN (nm)")
    ax.axvline(5.0, color="gray", ls=":", label="~5 nm")
    ax.set_xlabel(r"$d$ (u.a. ~ nm)"); ax.set_ylabel(r"$V_b$")
    ax.set_title("(c) Lei de Paschen vs ruptura por campo")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "paschen_fn_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Paschen / FN microplasma concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
