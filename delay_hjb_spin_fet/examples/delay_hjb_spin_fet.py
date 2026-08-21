#!/usr/bin/env python3
"""
Delay-HJB · Inércia de Spin · Spin-FET 2D (grafeno / WSe₂)
Capítulo 37 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.dinamica_spin_retardada import DinamicaSpinRetardada
from src.rede_pinn_hjb import RedePINN3D
from src.treinamento_hjb import treinar_delay_hjb
from src.hjbd_retardado import hamiltoniano_stt


def principal():
    print("=" * 70)
    print("  Delay-HJB · Spin-FET 2D · τ_spin (Cap. 37)")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    # dinâmica com retardo
    print("\n[1] Simulando spin com inércia (τ_spin)...")
    spin = DinamicaSpinRetardada(tau=0.15, gamma=0.8, beta_mem=0.4, alpha_stt=1.0, sigma=0.08, M0=0.0, dt=0.01, semente=42)

    def u_open_loop(M, M_tau, t):
        # pulso STT simples
        return 1.2 if 0.2 < t < 0.8 else 0.0

    traj = spin.simular(n_passos=1500, politica_u=u_open_loop)
    print(f"    M final={traj['M'][-1]:.3f}, M_tau final={traj['M_tau'][-1]:.3f}")

    # colocation no espaço estendido (x,y,t)
    print("\n[2] Pontos de colocation (M_t, M_{t−τ}, t)...")
    g = np.random.default_rng(0)
    n_col = 400
    X_col = np.column_stack([
        g.uniform(-1, 1, n_col),
        g.uniform(-1, 1, n_col),
        g.uniform(0, 1.5, n_col),
    ])
    # condição terminal V(x,y,T) ≈ (x − alvo)²
    n_term = 80
    X_term = np.column_stack([
        g.uniform(-1, 1, n_term),
        g.uniform(-1, 1, n_term),
        np.full(n_term, 1.5),
    ])
    V_term = (X_term[:, 0] - 0.7) ** 2

    print("\n[3] Treinando PINN Delay-HJB...")
    rede = RedePINN3D(camadas=[3, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_delay_hjb(
        rede, X_col, X_term, V_term,
        n_epocas=300, taxa=7e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # política ótima aproximada a partir de V_x
    print("\n[4] Política STT a partir de ∇V...")
    def u_pinn(M, M_tau, t):
        X = np.array([[M, M_tau, t]])
        _, _, Vx, _ = rede.derivadas(X)
        u = - (1.0 / (2.0 * 0.15)) * float(np.asarray(Vx).reshape(-1)[0])
        return float(np.clip(u, -2, 2))

    spin2 = DinamicaSpinRetardada(tau=0.15, gamma=0.8, beta_mem=0.4, alpha_stt=1.0, sigma=0.08, M0=0.0, dt=0.01, semente=99)
    traj2 = spin2.simular(n_passos=1500, politica_u=u_pinn)

    print("\n[5] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["M"], "C0-", lw=1.2, label=r"$M_t$ (pulso aberto)")
    ax.plot(traj["t"], traj["M_tau"], "C1--", lw=1.0, alpha=0.8, label=r"$M_{t-\tau}$")
    ax.set_xlabel(r"$t$")
    ax.set_title("(a) Dinâmica com inércia de spin")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj2["t"], traj2["M"], "C2-", lw=1.2, label=r"$M_t$ (STT PINN)")
    ax.plot(traj2["t"], traj2["u"], "C3--", lw=1.0, alpha=0.8, label=r"$u^*(t)$")
    ax.set_xlabel(r"$t$")
    ax.set_title("(b) Controle STT ótimo aproximado")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # mapa V(x, y=0, t=0.5)
    ax = eixos[1, 0]
    xs = np.linspace(-1, 1, 40)
    ys = np.linspace(-1, 1, 40)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, 0.5)])
    VV = rede.prever(pts).reshape(XX.shape)
    cf = ax.contourf(XX, YY, VV, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$M_t$")
    ax.set_ylabel(r"$M_{t-\tau}$")
    ax.set_title(r"(c) $V(M_t, M_{t-\tau}, t=0.5)$")

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época")
    ax.set_ylabel("perda Delay-HJB")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "delay_hjb_spin_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação Delay-HJB Spin-FET concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
