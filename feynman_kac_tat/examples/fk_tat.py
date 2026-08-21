#!/usr/bin/env python3
"""
Feynman–Kac com Saltos · TAT em HfO₂/ZrO₂
Cap. 17 & Apêndice A.7 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.processo_saltos import simular_trajetorias, densidade_salto_lognormal
from src.tat_dieletrico import DieletricoTAT
from src.rede_pinn_fk import RedePINN_FK
from src.treinamento_fk import treinar_fk


def principal():
    print("=" * 70)
    print("  Feynman–Kac + Saltos · TAT · HfO₂/ZrO₂ 1.6 nm")
    print("  Cap. 17 & A.7 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    diel = DieletricoTAT(lambda_salto=1.0, sigma_termico=0.28, r_relax=0.08)
    print(f"\n[1] Dielétrico TAT: λ={diel.lambda_salto}, σ={diel.sigma_termico}, κ={diel.kappa():.3f}")

    print("\n[2] Trajetórias jump-diffusion (Poole–Frenkel + saltos)...")
    traj = simular_trajetorias(
        n_traj=150, n_passos=250, dt=0.01, S0=1.0,
        r=diel.r_relax, sigma=diel.sigma_termico, lam=diel.lambda_salto, semente=42,
    )

    print("\n[3] PINN Feynman–Kac (PIDE com integral MC)...")
    g = np.random.default_rng(0)
    n_col = 300
    # S em log-espaço aproximadamente [0.3, 3]
    X_col = np.column_stack([
        g.uniform(0.3, 3.0, n_col),
        g.uniform(0.0, 1.0, n_col),
    ])
    # condição terminal V(S,T) = payoff / ocupação residual
    n_term = 60
    S_term = g.uniform(0.3, 3.0, n_term)
    X_term = np.column_stack([S_term, np.full(n_term, 1.0)])
    V_term = np.maximum(S_term - 1.0, 0.0)  # análogo a call / captura acima do nível

    rede = RedePINN_FK(camadas=[2, 28, 28, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_fk(
        rede, X_col, X_term, V_term,
        r=diel.r_relax, sigma=diel.sigma_termico, lam=diel.lambda_salto,
        n_epocas=250, taxa=7e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfil V(S, t=0.5)
    Sg = np.linspace(0.3, 3.0, 80)
    pts = np.column_stack([Sg, np.full_like(Sg, 0.5)])
    V_mid = rede.prever(pts)
    pts_T = np.column_stack([Sg, np.full_like(Sg, 1.0)])
    V_T = rede.prever(pts_T)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    for i in range(min(40, traj["S"].shape[0])):
        ax.plot(traj["t"], traj["S"][i], color="C0", alpha=0.25, lw=0.7)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$S_t$")
    ax.set_title("(a) Trajetórias (difusão + saltos TAT)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    eta = np.linspace(0.2, 2.5, 100)
    ax.plot(eta, densidade_salto_lognormal(eta, diel.mu_j, diel.sig_j), "k-", lw=2)
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$g(\eta)$")
    ax.set_title("(b) Densidade de salto (tunelamento entre armadilhas)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(Sg, V_mid, "C0-", lw=2, label=r"$V(S,0.5)$")
    ax.plot(Sg, V_T, "C3--", lw=2, label=r"$V(S,T)$")
    ax.plot(Sg, np.maximum(Sg - 1.0, 0.0), "k:", lw=1.5, label="terminal")
    ax.set_xlabel(r"$S$"); ax.set_ylabel(r"$V$")
    ax.set_title("(c) Solução PINN da PIDE Feynman–Kac")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "fk_tat_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação Feynman–Kac TAT concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
