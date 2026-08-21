#!/usr/bin/env python3
"""
Contágio Térmico McKean–Vlasov · 3D-IC / GAAFET
Capítulos 24 & 40 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.mckean_vlasov import simular_populacao
from src.rede_pinn_mv import RedePINN_MV
from src.treinamento_mv import treinar_mv
from src.residuo_mv import media_populacional


def principal():
    print("=" * 70)
    print("  McKean–Vlasov · Contágio Térmico · 3D-IC / GAAFET")
    print("  Cap. 24 & 40 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando população de nanotransistores (temperaturas)...")
    # sem runaway
    pop_ok = simular_populacao(
        n_particulas=250, n_passos=1200, dt=0.01,
        a=1.2, sigma=0.12, aquecimento=0.02, T_crit=1.8,
        T0_mean=0.7, T0_std=0.12, semente=42,
    )
    # com runaway
    pop_run = simular_populacao(
        n_particulas=250, n_passos=1200, dt=0.01,
        a=1.2, sigma=0.12, aquecimento=0.12, T_crit=1.0,
        T0_mean=0.9, T0_std=0.15, semente=7,
    )
    print(f"    Caso estável: T̄ final={pop_ok['mean'][-1]:.3f}")
    print(f"    Caso runaway: T̄ final={pop_run['mean'][-1]:.3f}")

    print("\n[2] PINN McKean–Vlasov p(x,t)...")
    g = np.random.default_rng(0)
    x_grade = np.linspace(0.2, 2.5, 80)
    n_col = 350
    X_col = np.column_stack([
        g.uniform(0.2, 2.5, n_col),
        g.uniform(0.0, 1.5, n_col),
    ])
    n0 = 70
    x0 = g.uniform(0.2, 2.5, n0)
    X0 = np.column_stack([x0, np.zeros(n0)])
    p0 = np.exp(-0.5 * ((x0 - 0.8) / 0.2) ** 2)

    rede = RedePINN_MV(camadas=[2, 28, 28, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_mv(
        rede, X_col, X0, p0, x_grade,
        a=1.2, sigma=0.12, n_epocas=280, taxa=6e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # evolução da média via PINN
    ts = np.linspace(0, 1.5, 30)
    Xbar_pinn = [media_populacional(rede, t, x_grade) for t in ts]

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(pop_ok["t"], pop_ok["mean"], "C0-", lw=2, label=r"$\bar T$ estável")
    ax.fill_between(pop_ok["t"], pop_ok["mean"] - pop_ok["std"], pop_ok["mean"] + pop_ok["std"],
                     color="C0", alpha=0.2)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"temperatura")
    ax.set_title("(a) População GAAFET — regime estável")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(pop_run["t"], pop_run["mean"], "C3-", lw=2, label=r"$\bar T$ runaway")
    ax.fill_between(pop_run["t"], pop_run["mean"] - pop_run["std"], pop_run["mean"] + pop_run["std"],
                     color="C3", alpha=0.2)
    ax.axhline(1.0, color="k", ls="--", alpha=0.5, label=r"$T_{\mathrm{crit}}$")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"temperatura")
    ax.set_title("(b) Fuga térmica em cascata")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(ts, Xbar_pinn, "C2-", lw=2, label=r"$\bar X_t$ via PINN")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\bar X_t$")
    ax.set_title(r"(c) Média populacional $\int x\,p_\theta(x,t)\,dx$")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda MV")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "mv_termico_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação McKean–Vlasov térmica concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
