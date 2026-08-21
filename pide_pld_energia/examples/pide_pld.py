#!/usr/bin/env python3
"""
PIDE · Opções sobre PLD / Swing de energia · ACL/CCEE
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.pld_hidrologia import simular_pld, theta_sazonal
from src.rede_pinn_pide import RedePINN_PIDE
from src.treinamento_pide import treinar_pide


def principal():
    print("=" * 70)
    print("  PIDE · Derivativos de Energia · Opções sobre PLD")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando PLD com sazonalidade e saltos de ENA...")
    traj = simular_pld(n_passos=800, dt=0.01, S0=180.0, k=2.0, sigma=0.5, lam=1.2, semente=42)
    print(f"    PLD: média={traj['S'].mean():.1f}, max={traj['S'].max():.1f} R$/MWh")

    print("\n[2] PINN PIDE (call europeia / swing simplificado)...")
    g = np.random.default_rng(0)
    # S normalizado: log-scale ~ [50, 800] R$/MWh → usamos S em unidades
    n_col = 300
    S_col = g.uniform(50, 600, n_col) / 200.0  # normalizado
    X_col = np.column_stack([
        S_col,
        g.uniform(0.0, 1.0, n_col),
    ])
    # terminal: payoff call K=200
    K = 200.0
    n_term = 80
    S_term_raw = g.uniform(50, 600, n_term)
    S_term = S_term_raw / 200.0
    X_term = np.column_stack([S_term, np.ones(n_term)])
    V_term = np.maximum(S_term_raw - K, 0.0) / 200.0  # payoff normalizado

    rede = RedePINN_PIDE(camadas=[2, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_pide(
        rede, X_col, X_term, V_term,
        n_epocas=250, taxa=8e-4, semente=0, verbose_cada=50,
        k=1.5, sigma=0.3, r=0.08, lam=0.8,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    Sg = np.linspace(50, 500, 80)
    V0 = rede.prever(np.column_stack([Sg/200.0, np.zeros_like(Sg)])) * 200.0
    VT = rede.prever(np.column_stack([Sg/200.0, np.ones_like(Sg)])) * 200.0
    payoff = np.maximum(Sg - K, 0.0)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["S"], "C0-", lw=1.0, alpha=0.8, label="PLD")
    ax.plot(traj["t"], traj["theta"], "C3--", lw=1.5, label=r"$e^{\theta(t)}$ sazonal")
    ax.set_xlabel(r"$t$ (anos)"); ax.set_ylabel(r"R\$/MWh")
    ax.set_title("(a) PLD com sazonalidade e saltos de hidrologia")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    tg = np.linspace(0, 1, 100)
    ax.plot(tg, [theta_sazonal(t) for t in tg], "k-", lw=2)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\theta(t)=\ln$ nível")
    ax.set_title("(b) Reversão à média sazonal")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(Sg, payoff, "k:", lw=1.5, label=r"payoff $(S-K)^+$")
    ax.plot(Sg, VT, "C3--", lw=2, label=r"$V(S,T)$ PINN")
    ax.plot(Sg, V0, "C0-", lw=2, label=r"$V(S,0)$ PINN")
    ax.set_xlabel(r"PLD $S$"); ax.set_ylabel(r"$V$")
    ax.set_title("(c) Opção sobre PLD / Swing (payoff)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda PIDE")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "pide_pld_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  PIDE PLD / energia concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
