#!/usr/bin/env python3
"""
Fokker–Planck + Saltos Discretos · Memória de Elétron Único (SET)
Capítulo 8 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.set_carga import simular_set, taxas_tunelamento
from src.rede_pinn_set import RedePINN_SET
from src.treinamento_set import treinar_set


def principal():
    print("=" * 70)
    print("  FP + Saltos · SET / Memória de Elétron Único")
    print("  Capítulo 8 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando dinâmica híbrida q (discreto) + s (contínuo)...")
    traj = simular_set(n_passos=4000, dt=0.01, q0=0, s0=0.1, sigma=0.12, E_c=0.5, V_bias=0.4, semente=42)
    print(f"    q: min={traj['q'].min()}, max={traj['q'].max()}, final={traj['q'][-1]}")
    print(f"    s: média={traj['s'].mean():.3f}")

    # taxas vs s
    sg = np.linspace(-0.5, 1.5, 80)
    la, lb = taxas_tunelamento(sg, q=0, Gamma0=1.0, E_c=0.5, V_bias=0.4)

    print("\n[2] PINN p(q,s,t)...")
    g = np.random.default_rng(0)
    n_col = 350
    q = g.integers(-2, 4, n_col).astype(float)
    s = g.uniform(-0.5, 1.5, n_col)
    t = g.uniform(0, 1.5, n_col)
    n0 = 60
    q0 = g.integers(-1, 2, n0).astype(float)
    s0 = g.uniform(-0.5, 1.5, n0)
    p0 = np.exp(-0.5 * ((q0 - 0) / 1.0) ** 2 - 0.5 * ((s0 - 0.2) / 0.3) ** 2)

    rede = RedePINN_SET(camadas=[3, 28, 28, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_set(
        rede, q, s, t, q0, s0, p0, sigma=0.12,
        n_epocas=250, taxa=7e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # mapa p(q,s) em t fixo
    qg = np.arange(-2, 5)
    sg2 = np.linspace(-0.5, 1.5, 40)
    QQ, SS = np.meshgrid(qg, sg2, indexing="ij")
    pts = np.column_stack([QQ.ravel() / 5.0, SS.ravel(), np.full(QQ.size, 1.0)])
    P = rede.prever(pts).reshape(QQ.shape)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.step(traj["t"], traj["q"], where="post", color="C0", lw=1.0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$q$ (elétrons)")
    ax.set_title("(a) Número discreto de elétrons (degraus de Coulomb)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj["t"], traj["s"], "C3-", lw=0.8, alpha=0.8)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$s$ (potencial)")
    ax.set_title("(b) Potencial eletrostático contínuo")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(sg, la, "C0-", lw=2, label=r"$\lambda^a$ (adição)")
    ax.plot(sg, lb, "C3-", lw=2, label=r"$\lambda^b$ (remoção)")
    ax.set_xlabel(r"$s$"); ax.set_ylabel("taxa")
    ax.set_title("(c) Taxas de tunelamento")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    im = ax.imshow(P, aspect="auto", origin="lower",
                   extent=[sg2[0], sg2[-1], qg[0] - 0.5, qg[-1] + 0.5],
                   cmap="magma")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$s$"); ax.set_ylabel(r"$q$")
    ax.set_title(r"(d) $p_\theta(q,s,t=1)$ PINN")

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "fp_set_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    # histórico
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.semilogy(res["historico"], "C4-", lw=1.5)
    ax2.set_xlabel("época"); ax2.set_ylabel("perda")
    ax2.set_title("Treinamento PINN Kolmogorov")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "fp_set_treino.png"), dpi=120)

    print("\n" + "=" * 70)
    print("  Simulação FP–SET concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
