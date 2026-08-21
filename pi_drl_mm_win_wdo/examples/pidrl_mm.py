#!/usr/bin/env python3
"""
PI-DRL · Market Making WIN / WDO · Avellaneda–Stoikov
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.avellaneda_stoikov import simular_mm, reservas_as, intensidade_chegada
from src.rede_pidrl import Critic, Actor
from src.treinamento_pidrl import treinar_pidrl, coletar_experiencia


def principal():
    print("=" * 70)
    print("  PI-DRL · Market Making Mini-Índice (WIN) / Mini-Dólar (WDO)")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando market maker AS (baseline)...")
    traj = simular_mm(n_passos=2500, dt=0.01, s0=120000, sigma=80.0, gamma=0.01, A=1.5, k=0.8, semente=42)
    # escala tipo WIN points
    print(f"    PnL final={traj['pnl'][-1]:.1f}, q final={traj['q'][-1]:.0f}")

    print("\n[2] Treinando Critic PI-DRL (HJB regularizado)...")
    critic = Critic(semente=42)
    actor = Actor(semente=43)
    print(f"    Critic params: {critic.n_parametros()}, Actor: {actor.n_parametros()}")
    res = treinar_pidrl(
        critic, actor, n_epocas=200, taxa_c=1e-3, semente=0, verbose_cada=40,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # política AS vs inventário
    qs = np.linspace(-8, 8, 50)
    deltas = []
    for q in qs:
        _, d = reservas_as(100.0, q, sigma=0.5, gamma=0.1, T_resto=1.0, k=1.0)
        deltas.append(d)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["s"], "C0-", lw=0.8, alpha=0.8)
    ax.set_xlabel(r"$t$"); ax.set_ylabel("mid (pts)")
    ax.set_title("(a) Mid price WIN/WDO (simulado)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj["t"], traj["q"], "C3-", lw=1.0)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"inventário $q$")
    ax.set_title("(b) Inventário (evitar overshoot)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(traj["t"], traj["pnl"], "C2-", lw=1.2)
    ax.set_xlabel(r"$t$"); ax.set_ylabel("PnL")
    ax.set_title("(c) PnL do market maker AS")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.plot(qs, deltas, "k-", lw=2)
    ax.set_xlabel(r"$q$"); ax.set_ylabel(r"$\delta^*$")
    ax.set_title("(d) Spread ótimo AS vs inventário")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "pidrl_mm_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.semilogy(np.maximum(res["historico"], 1e-12), "C4-", lw=1.5)
    ax2.set_xlabel("época"); ax2.set_ylabel("perda Critic (TD+HJB)")
    ax2.set_title("Treinamento PI-DRL")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "pidrl_treino.png"), dpi=120)

    print("\n" + "=" * 70)
    print("  PI-DRL Market Making concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
