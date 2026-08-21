#!/usr/bin/env python3
"""
Heston PINN · PETR4 / VALE3 · Preço + Gregas (Δ, Γ, Vanna)
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.heston import simular_heston, payoff_call
from src.rede_pinn_heston import RedePINN_Heston
from src.treinamento_heston import treinar_heston


def principal():
    print("=" * 70)
    print("  Heston PINN · PETR4 / VALE3 · B3")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    # parâmetros estilo PETR4 (commodities, skew negativo)
    r, kappa, theta, xi, rho = 0.10, 2.0, 0.04, 0.45, -0.75
    S0, K = 35.0, 35.0
    print(f"\n[1] Heston: ρ={rho}, ξ={xi}, κ={kappa}, θ={theta}")

    print("\n[2] Simulando trajetórias S,v (PETR4-like)...")
    traj = simular_heston(
        n_passos=600, dt=0.002, S0=S0, v0=0.05,
        r=r, kappa=kappa, theta=theta, xi=xi, rho=rho, semente=42,
    )
    print(f"    S final={traj['S'][-1]:.2f}, v médio={traj['v'].mean():.4f}")

    print("\n[3] PINN V(S,v,τ)...")
    g = np.random.default_rng(0)
    n_col = 400
    # S em [15, 60], v em [0.01, 0.25], τ em [0.05, 1.0]
    X_col = np.column_stack([
        g.uniform(15, 60, n_col),
        g.uniform(0.01, 0.25, n_col),
        g.uniform(0.05, 1.0, n_col),
    ])
    n_term = 80
    S_term = g.uniform(15, 60, n_term)
    v_term = g.uniform(0.01, 0.25, n_term)
    X_term = np.column_stack([S_term, v_term, np.zeros(n_term)])
    V_term = payoff_call(S_term, K)

    rede = RedePINN_Heston(camadas=[3, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_heston(
        rede, X_col, X_term, V_term,
        n_epocas=280, taxa=7e-4, semente=0, verbose_cada=50,
        r=r, kappa=kappa, theta=theta, xi=xi, rho=rho,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # superfície e gregas em τ=0.5, v=0.04
    Sg = np.linspace(20, 55, 50)
    Xg = np.column_stack([Sg, np.full_like(Sg, 0.04), np.full_like(Sg, 0.5)])
    gregs = rede.gregas(Xg)
    payoff = payoff_call(Sg, K)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["S"], "C0-", lw=1.0, alpha=0.85)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S_t$ (R\$)")
    ax.set_title("(a) PETR4-like sob Heston")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj["t"], np.sqrt(traj["v"]) * 100, "C3-", lw=1.0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"vol instantânea (%)")
    ax.set_title(r"(b) $\sqrt{v_t}$ (vol estocástica)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(Sg, payoff, "k:", lw=1.5, label="payoff")
    ax.plot(Sg, gregs["V"], "C0-", lw=2, label=r"$V_{\mathrm{NN}}$")
    ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$V$")
    ax.set_title(r"(c) Preço call Heston ($\tau=0.5$)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.plot(Sg, gregs["Delta"], "C0-", lw=2, label=r"$\Delta$")
    ax.plot(Sg, gregs["Gamma"] * 10, "C3-", lw=2, label=r"$10\times\Gamma$")
    ax.plot(Sg, gregs["Vanna"], "C2--", lw=1.5, label="Vanna")
    ax.set_xlabel(r"$S$"); ax.set_ylabel("gregas")
    ax.set_title(r"(d) $\Delta$, $\Gamma$, Vanna (hedging)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "heston_petr4_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.semilogy(np.maximum(res["historico"], 1e-12), "C4-", lw=1.5)
    ax2.set_xlabel("época"); ax2.set_ylabel("perda")
    ax2.set_title("Treinamento PINN Heston")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "heston_treino.png"), dpi=120)

    print("\n" + "=" * 70)
    print("  Heston PINN PETR4/VALE3 concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
