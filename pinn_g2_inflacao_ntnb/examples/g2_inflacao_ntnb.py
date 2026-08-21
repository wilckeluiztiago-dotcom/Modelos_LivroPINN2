#!/usr/bin/env python3
"""
PINN G2++ · Inflação Implícita (NTN-B vs DI) · Curva Real
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.modelo_g2_inflacao import (
    simular_fatores,
    preco_nominal_analitico_approx,
    preco_real_analitico_approx,
    inflacao_implicita_breakeven,
)
from src.rede_pinn_g2 import RedePINN_G2
from src.treinamento_g2 import treinar_g2


def principal():
    print("=" * 70)
    print("  PINN G2++ · NTN-B vs DI · Inflação Implícita")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando fatores r_t (real) e i_t (inflação)...")
    traj = simular_fatores(n_passos=800, dt=0.02, r0=0.04, i0=0.045, semente=42)
    print(f"    r médio={traj['r'].mean():.4f}, i médio={traj['i'].mean():.4f}")

    print("\n[2] PINN 3D P(r, i, τ)...")
    g = np.random.default_rng(0)
    n_col = 400
    X_col = np.column_stack([
        g.uniform(0.01, 0.10, n_col),   # r
        g.uniform(0.01, 0.10, n_col),   # i
        g.uniform(0.1, 10.0, n_col),    # τ
    ])
    n_term = 80
    X_term = np.column_stack([
        g.uniform(0.01, 0.10, n_term),
        g.uniform(0.01, 0.10, n_term),
        np.zeros(n_term),
    ])

    rede = RedePINN_G2(camadas=[3, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_g2(
        rede, X_col, X_term,
        n_epocas=300, taxa=8e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # curvas de preço e breakeven ao longo de τ
    print("\n[3] Inflação implícita (breakeven)...")
    r0, i0 = 0.04, 0.045
    taus = np.linspace(0.5, 10, 40)
    P_nom = []
    P_real = []
    be = []
    for tau in taus:
        Xn = np.array([[r0, i0, tau]])
        pn = float(rede.prever(Xn))
        # preço real: mesma rede com i=0 efetivo (aprox) ou fórmula
        pr = preco_real_analitico_approx(r0, tau)
        P_nom.append(pn)
        P_real.append(pr)
        be.append(inflacao_implicita_breakeven(pn, pr, tau))
    P_nom, P_real, be = map(np.array, (P_nom, P_real, be))
    # referência Fisher
    be_fisher = np.full_like(taus, i0)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["r"] * 100, "C0-", lw=1.2, label=r"$r_t$ real")
    ax.plot(traj["t"], traj["i"] * 100, "C3-", lw=1.2, label=r"$i_t$ inflação")
    ax.plot(traj["t"], traj["n"] * 100, "C2--", lw=1.0, alpha=0.8, label=r"$n_t\approx r+i$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel("% a.a.")
    ax.set_title("(a) Fatores estocásticos G2++")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(taus, P_nom, "C0-", lw=2, label=r"$P_{\mathrm{nom}}$ (DI/LTN)")
    ax.plot(taus, P_real, "C3-", lw=2, label=r"$P_{\mathrm{real}}$ (NTN-B)")
    ax.set_xlabel(r"$\tau$ (anos)"); ax.set_ylabel("preço")
    ax.set_title("(b) Preços nominal vs real")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(taus, be * 100, "C0-", lw=2, label="breakeven PINN")
    ax.plot(taus, be_fisher * 100, "k--", lw=1.5, label=r"$i_0$ Fisher")
    ax.set_xlabel(r"$\tau$ (anos)"); ax.set_ylabel("% a.a.")
    ax.set_title("(c) Inflação implícita (livre de liquidez)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN 3D")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "g2_inflacao_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  PINN G2++ inflação implícita concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
