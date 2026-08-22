#!/usr/bin/env python3
"""
BTBT Kane/Keldysh · TFET · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_btbt import parametros_btbt_default, G_Kane, doping_tfet_1d
from src.rede_pinn_btbt import RedePINN_BTBT
from src.treinamento_btbt import treinar_btbt


def principal():
    print("=" * 70)
    print("  BTBT Kane · TFET · Tunelamento Interbandas · PINN")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_btbt_default()
    print(f"\n[1] Device: {dev}, E_g={p['E_g']}, A={p['A_Kane']}, B={p['B_Kane']}")

    # G vs E
    E_f = np.linspace(0.2, 8, 100)
    G = G_Kane(E_f, p)

    n_col = 250
    x_col = torch.linspace(0.02, 0.98, n_col, device=dev).reshape(-1, 1).requires_grad_(True)

    # BC: φ(0)=0 (fonte), φ(1)=Vds
    Vds = 1.0
    x_bc = torch.tensor([[0.0], [1.0]], device=dev)
    phi_bc = torch.tensor([[0.0], [Vds]], device=dev)

    print("\n[2] Treinando PINN (φ, n, p)...")
    rede = RedePINN_BTBT([1, 48, 48, 48, 3]).to(dev)
    res = treinar_btbt(
        rede, x_col, x_bc, phi_bc, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    xg = torch.linspace(0, 1, 150, device=dev).reshape(-1, 1)
    xg.requires_grad_(True)
    phi, n, p_h = rede.campos(xg)
    dphi = torch.autograd.grad(phi, xg, torch.ones_like(phi), create_graph=False)[0]
    x_np = xg.detach().cpu().numpy().ravel()
    phi_np = phi.detach().cpu().numpy().ravel()
    n_np = n.detach().cpu().numpy().ravel()
    p_np = p_h.detach().cpu().numpy().ravel()
    E_np = dphi.detach().cpu().numpy().ravel()
    G_np = G_Kane(E_np, p)
    N_net = doping_tfet_1d(x_np, p)

    # SS proxy: log J vs V seria ideal; aqui mostramos G integrado
    print(f"    G_BTBT max = {G_np.max():.4e}")

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x_np, phi_np, "C0-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi$")
    ax.set_title(r"(a) Potencial $\phi$ (TFET $V_{ds}$)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(x_np, n_np, "C0-", lw=2, label=r"$n$")
    ax.plot(x_np, p_np, "C3-", lw=2, label=r"$p$")
    ax.plot(x_np, N_net, "k--", lw=1, alpha=0.6, label=r"$N_{net}$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel("densidade")
    ax.set_title("(b) Portadores e dopagem (p+|i|n+)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.semilogy(E_f, np.maximum(G, 1e-12), "C2-", lw=2)
    ax.set_xlabel(r"$|\mathcal{E}|$"); ax.set_ylabel(r"$G_{BTBT}$")
    ax.set_title(r"(c) Kane $G_{BTBT}(\mathcal{E})$")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.plot(x_np, G_np, "C3-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$G_{BTBT}(x)$")
    ax.set_title(r"(d) Geração BTBT ao longo do canal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "btbt_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.semilogy(res["historico"], "C4-", lw=1.2)
    ax2.set_xlabel("época"); ax2.set_ylabel("perda")
    ax2.set_title("Treinamento PINN BTBT")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "btbt_treino.png"), dpi=120)

    print("\n" + "=" * 70)
    print("  BTBT TFET concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
