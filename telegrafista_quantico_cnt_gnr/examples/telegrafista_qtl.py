#!/usr/bin/env python3
"""
Telegrafista Quântico · CNT / GNR · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_telegrafista import (
    parametros_qtl_default, impedancia_caracteristica, velocidade_onda,
)
from src.rede_pinn_qtl import RedePINN_QTL
from src.treinamento_qtl import treinar_qtl


def principal():
    print("=" * 70)
    print("  Telegrafista Quântico · CNT / GNR · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_qtl_default()
    Z0 = impedancia_caracteristica(p)
    v = velocidade_onda(p)
    print(f"\n[1] Device: {dev}")
    print(f"    L_K={p['L_K']}, C_Q={p['C_Q']}, L_tot={p['L_tot']:.3f}, C_eff={p['C_eff']:.3f}")
    print(f"    Z0≈{Z0:.3f}, v≈{v:.3f}")

    # colocation
    n_col = 500
    z = torch.rand(n_col, 1, device=dev)
    t = torch.rand(n_col, 1, device=dev)
    zt_col = torch.cat([z, t], dim=1).requires_grad_(True)

    # fonte z=0: pulso suave V_src(t) = sin(π t) para t in [0,1]
    n_src = 60
    t_src = torch.linspace(0, 1, n_src, device=dev).reshape(-1, 1)
    z_src = torch.zeros(n_src, 1, device=dev)
    zt_src = torch.cat([z_src, t_src], dim=1)
    V_src = torch.sin(np.pi * t_src)

    # carga z=1: V = Z_L I
    n_load = 40
    t_load = torch.linspace(0, 1, n_load, device=dev).reshape(-1, 1)
    z_load = torch.ones(n_load, 1, device=dev)
    zt_load = torch.cat([z_load, t_load], dim=1)

    print("\n[2] Treinando PINN (V, I)...")
    rede = RedePINN_QTL([2, 48, 48, 48, 2]).to(dev)
    npar = sum(q.numel() for q in rede.parameters())
    print(f"    Parâmetros: {npar}")
    res = treinar_qtl(
        rede, zt_col, zt_src, V_src, zt_load, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # mapa V(z,t)
    nz, nt = 80, 50
    Z, T = np.meshgrid(np.linspace(0, 1, nz), np.linspace(0, 1, nt))
    zt_m = torch.tensor(np.column_stack([Z.ravel(), T.ravel()]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        Vm, Im = rede.campos(zt_m)
    Vm = Vm.cpu().numpy().reshape(nt, nz)
    Im = Im.cpu().numpy().reshape(nt, nz)

    # perfil em t=0.5
    zg = torch.linspace(0, 1, 100, device=dev).reshape(-1, 1)
    zt05 = torch.cat([zg, torch.full_like(zg, 0.5)], dim=1)
    with torch.no_grad():
        V05, I05 = rede.campos(zt05)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    im = ax.contourf(Z, T, Vm, levels=20, cmap="RdBu_r")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(a) $V(z,t)$ — onda na linha quântica")

    ax = eixos[0, 1]
    im = ax.contourf(Z, T, Im, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(b) $I(z,t)$")

    ax = eixos[1, 0]
    z_np = zg.cpu().numpy().ravel()
    ax.plot(z_np, V05.cpu().numpy().ravel(), "C0-", lw=2, label=r"$V$")
    ax.plot(z_np, I05.cpu().numpy().ravel(), "C3-", lw=2, label=r"$I$")
    ax.set_xlabel(r"$z$"); ax.set_title(r"(c) Perfil em $t=0.5$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "qtl_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Telegrafista quântico CNT/GNR concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
