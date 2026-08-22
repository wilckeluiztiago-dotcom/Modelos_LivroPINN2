#!/usr/bin/env python3
"""
Landauer–Büttiker · Condutância Quantizada · Constrições 1D · PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_landauer import (
    modos_analiticos_poco, condutancia_vs_gate, parametros_lb_default,
)
from src.rede_pinn_modos import BancoModos
from src.treinamento_modos import treinar_modos


def principal():
    print("=" * 70)
    print("  Landauer–Büttiker · Condutância Quantizada · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    W = 1.0
    n_modos = 3
    print(f"\n[1] Device: {dev}, W={W}, n_modos={n_modos}")

    # referência analítica
    E_ref = [modos_analiticos_poco(n, W)[0] for n in range(1, n_modos + 1)]
    print(f"    E_n analítico: {[round(e, 4) for e in E_ref]}")

    # pontos de colocation interior (0,W)
    n_col = 120
    y_col = torch.linspace(0.02, W - 0.02, n_col, device=dev).reshape(-1, 1).requires_grad_(True)
    y_quad = torch.linspace(0.0, W, 200, device=dev).reshape(-1, 1)

    print("\n[2] Treinando modos transversais (Schrödinger + ortogonalidade)...")
    banco = BancoModos(n_modos=n_modos, camadas=[1, 48, 48, 1]).to(dev)
    res = treinar_modos(banco, y_col, y_quad, W, n_epocas=3000, taxa=1e-3, verbose_cada=300)
    print(f"    Perda final: {res['perda_final']:.4e}")
    E_nn = banco.energias().detach().cpu().numpy()
    print(f"    E_n PINN:    {E_nn.round(4)}")

    # funções de onda
    y_plot = torch.linspace(0, W, 200, device=dev).reshape(-1, 1)
    with torch.no_grad():
        psis = [banco.psi(n, y_plot).cpu().numpy().ravel() for n in range(n_modos)]
    y_np = y_plot.cpu().numpy().ravel()
    psis_ref = [modos_analiticos_poco(n, W)[1](y_np) for n in range(1, n_modos + 1)]

    # condutância
    Vg = np.linspace(0, 25, 100)
    cond = condutancia_vs_gate(Vg, n_modos=5, W=W)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    for n in range(n_modos):
        # alinhar sinal
        s = np.sign(np.trapezoid(psis[n] * psis_ref[n], y_np)) or 1
        ax.plot(y_np, s * psis[n], lw=2, label=fr"$\psi_{n+1}$ PINN")
        ax.plot(y_np, psis_ref[n], "k--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$y$"); ax.set_ylabel(r"$\psi_n(y)$")
    ax.set_title("(a) Modos transversais (PINN vs analítico)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ns = np.arange(1, n_modos + 1)
    ax.bar(ns - 0.15, E_ref, 0.3, label="analítico", color="C0")
    ax.bar(ns + 0.15, E_nn, 0.3, label="PINN", color="C3")
    ax.set_xlabel(r"$n$"); ax.set_ylabel(r"$E_n$")
    ax.set_title("(b) Energias dos subníveis")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.step(cond["V_g"], cond["G_sobre_G0"], where="mid", color="C2", lw=2)
    ax.set_xlabel(r"$V_g$ efetivo"); ax.set_ylabel(r"$G/G_0$")
    ax.set_title(r"(c) Condutância quantizada ($G_0=2e^2/h$)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "landauer_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Landauer–Büttiker concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
