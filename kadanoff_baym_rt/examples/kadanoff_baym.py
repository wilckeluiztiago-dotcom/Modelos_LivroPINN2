#!/usr/bin/env python3
"""
Kadanoff–Baym em Tempo Real · GW/Fock · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_kb import parametros_kb_default
from src.rede_pinn_kb import RedePINN_KB
from src.treinamento_kb import treinar_kb


def principal():
    print("=" * 70)
    print("  Kadanoff–Baym RT · G^</G^R · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_kb_default()
    print(f"\n[1] Device: {dev}, ε0={p['eps0']}, U_HF={p['U_HF']}, γ={p['gamma']}")

    n_col = 400
    t_max = p["t_max"]
    t1 = torch.rand(n_col, 1, device=dev) * t_max
    t2 = torch.rand(n_col, 1, device=dev) * t_max
    t12 = torch.cat([t1, t2], dim=1).requires_grad_(True)
    t12_swap = torch.cat([t2.detach(), t1.detach()], dim=1)

    print("\n[2] Treinando PINN G^<(t1,t2), G^R(t1,t2)...")
    rede = RedePINN_KB([2, 48, 48, 48, 4]).to(dev)
    res = treinar_kb(rede, t12, t12_swap, p, n_epocas=2500, taxa=1e-3, verbose_cada=250)
    print(f"    Perda final: {res['perda_final']:.4e}")

    # mapa G^< no plano (t1,t2)
    N = 60
    tg = np.linspace(0, t_max, N)
    T1, T2 = np.meshgrid(tg, tg)
    pts = torch.tensor(np.column_stack([T1.ravel(), T2.ravel()]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Gl_R, Gl_I = rede.G_lesser(pts)
        Gr_R, Gr_I = rede.G_retarded(pts)
    Gl_map = Gl_R.cpu().numpy().reshape(N, N)
    Gr_map = Gr_R.cpu().numpy().reshape(N, N)

    # diagonal t1=t2: ocupação ~ −i G^<(t,t)
    td = torch.linspace(0, t_max, 100, device=dev).reshape(-1, 1)
    tdd = torch.cat([td, td], dim=1)
    with torch.no_grad():
        gR, gI = rede.G_lesser(tdd)
    # n(t) ~ −Im G^< ou −i G^< para orbital
    n_t = -gI.cpu().numpy().ravel()

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    im = ax.contourf(T1, T2, Gl_map, levels=20, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$t_1$"); ax.set_ylabel(r"$t_2$")
    ax.set_title(r"(a) $\mathrm{Re}\,G^<(t_1,t_2)$")

    ax = eixos[0, 1]
    im = ax.contourf(T1, T2, Gr_map, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$t_1$"); ax.set_ylabel(r"$t_2$")
    ax.set_title(r"(b) $\mathrm{Re}\,G^R(t_1,t_2)$ (causal)")

    ax = eixos[1, 0]
    ax.plot(td.cpu().numpy().ravel(), n_t, "C0-", lw=2)
    ax.axhline(p["n_eq"], color="k", ls="--", label=r"$n_{\mathrm{eq}}$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$-\mathrm{Im}\,G^<(t,t)$")
    ax.set_title(r"(c) Ocupação diagonal (transiente)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN KB")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "kb_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Kadanoff–Baym RT concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
