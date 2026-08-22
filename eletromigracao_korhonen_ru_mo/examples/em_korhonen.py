#!/usr/bin/env python3
"""
Eletromigração Atômica · Korhonen · Interconexões Ru/Mo · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_korhonen import parametros_korhonen_default
from src.rede_pinn_em import RedePotencial, RedeTensao
from src.treinamento_em import treinar_em


def principal():
    print("=" * 70)
    print("  Eletromigração Korhonen · Ru/Mo · 1 nm · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_korhonen_default()
    print(f"\n[1] Device: {dev}")
    print(f"    D_eff={p['D_eff']}, Z*e/Ω={p['Z_star_e_Omega']}")

    # φ: colocation + BC φ(0)=0, φ(1)=V
    n_phi = 80
    x_phi = torch.linspace(0, 1, n_phi, device=dev).reshape(-1, 1).requires_grad_(True)
    V_bias = 1.0
    x_bc_phi = torch.tensor([[0.0], [1.0]], device=dev)
    phi_bc_val = torch.tensor([[0.0], [V_bias]], device=dev)

    # σ_H: (x,t) colocation
    n_col = 400
    x_c = torch.rand(n_col, 1, device=dev)
    t_c = torch.rand(n_col, 1, device=dev) * 2.0
    xt_col = torch.cat([x_c, t_c], dim=1).requires_grad_(True)

    # fluxo BC em x=0,1 para vários t
    n_f = 40
    t_f = torch.rand(n_f, 1, device=dev) * 2.0
    x0 = torch.zeros(n_f, 1, device=dev)
    x1 = torch.ones(n_f, 1, device=dev)
    x_bc = torch.cat([x0, x1], dim=0).requires_grad_(True)
    t_bc = torch.cat([t_f, t_f], dim=0)

    # IC σ_H(x,0)=0
    n0 = 60
    x0_ic = torch.linspace(0, 1, n0, device=dev).reshape(-1, 1)
    t0_ic = torch.zeros(n0, 1, device=dev)
    sigma0_xt = torch.cat([x0_ic, t0_ic], dim=1)
    sigma0_val = torch.zeros(n0, 1, device=dev)

    print("\n[2] Treinando PINN (φ, σ_H)...")
    rede_phi = RedePotencial([1, 32, 32, 1]).to(dev)
    rede_sigma = RedeTensao([2, 48, 48, 1]).to(dev)
    npar = sum(q.numel() for q in list(rede_phi.parameters()) + list(rede_sigma.parameters()))
    print(f"    Parâmetros: {npar}")

    res = treinar_em(
        rede_sigma, rede_phi,
        x_phi, xt_col, x_bc, t_bc, phi_bc_val, sigma0_xt, sigma0_val, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfis
    xg = torch.linspace(0, 1, 120, device=dev).reshape(-1, 1)
    with torch.no_grad():
        phi = rede_phi(xg).cpu().numpy().ravel()
    x_np = xg.cpu().numpy().ravel()

    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x_np, phi, "C0-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi$")
    ax.set_title(r"(a) Potencial $\phi(x)$ (Ohm / Ru-Mo)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    for tval, style in [(0.5, "-"), (1.0, "--"), (2.0, ":")]:
        xt = torch.cat([xg, torch.full_like(xg, tval)], dim=1)
        with torch.no_grad():
            s = rede_sigma(xt).cpu().numpy().ravel()
        ax.plot(x_np, s, style, lw=2, label=fr"$t={tval}$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\sigma_H$")
    ax.set_title(r"(b) Tensão hidrostática (Korhonen)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    # mapa σ(x,t)
    nx, nt = 60, 40
    X, T = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 2, nt))
    xtm = torch.tensor(np.column_stack([X.ravel(), T.ravel()]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Sm = rede_sigma(xtm).cpu().numpy().reshape(nt, nx)
    im = ax.contourf(X, T, Sm, levels=20, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(c) $\sigma_H(x,t)$ — build-up de tensão")

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "em_korhonen_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"\n[3] Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Eletromigração Korhonen concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
