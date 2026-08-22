#!/usr/bin/env python3
"""
Kohn–Luttinger · EMA + Célula Central · ³¹P em Si · PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_ema import parametros_ema_default, E_BIND, V_total
from src.rede_pinn_ema import BancoEMA
from src.treinamento_ema import treinar_ema


def principal():
    print("=" * 70)
    print("  Kohn–Luttinger · ³¹P:Si · Valley-Orbit · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_ema_default()
    print(f"\n[1] Device: {dev}")
    print(f"    Eb exp: A1={E_BIND['A1']}, T2={E_BIND['T2']}, E={E_BIND['E']} meV")

    n_col = 150
    r_col = torch.linspace(0.05, 8.0, n_col, device=dev).reshape(-1, 1).requires_grad_(True)
    r_quad = torch.linspace(0.02, 10.0, 250, device=dev).reshape(-1, 1)

    print("\n[2] Treinando envelopes A1, T2, E...")
    banco = BancoEMA([1, 48, 48, 1]).to(dev)
    res = treinar_ema(banco, r_col, r_quad, p, n_epocas=3000, taxa=1e-3, verbose_cada=300)
    print(f"    Perda final: {res['perda_final']:.4e}")
    Ed = banco.energias_dict()
    print(f"    E PINN (norm): {Ed}")

    r_plot = torch.linspace(0.02, 6.0, 200, device=dev).reshape(-1, 1)
    with torch.no_grad():
        Fs = {s: banco.F(s, r_plot).cpu().numpy().ravel() for s in banco.SIMETRIAS}
    r_np = r_plot.cpu().numpy().ravel()
    V = V_total(r_np, p)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    for s, c in zip(banco.SIMETRIAS, ["C0", "C3", "C2"]):
        ax.plot(r_np, Fs[s], color=c, lw=2, label=s)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$F(r)$")
    ax.set_title(r"(a) Envelopes radiais (EMA + $V_{cc}$)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(r_np, V, "k-", lw=2)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$V(r)$")
    ax.set_title(r"(b) $V_{coul}+V_{cc}$")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    nomes = list(E_BIND.keys())
    exp = [E_BIND[s] for s in nomes]
    # reescala PINN para meV usando |E_A1|
    scale = E_BIND["A1"] / max(abs(Ed["A1"]), 1e-6)
    pin = [abs(Ed[s]) * scale for s in nomes]
    x = np.arange(len(nomes))
    ax.bar(x - 0.15, exp, 0.3, label="exp (meV)", color="C0")
    ax.bar(x + 0.15, pin, 0.3, label="PINN (esc.)", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(nomes)
    ax.set_ylabel(r"$E_b$ (meV)")
    ax.set_title(r"(c) Splitting valley-orbit $1s(A_1,T_2,E)$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "kl_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Kohn–Luttinger ³¹P:Si concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
