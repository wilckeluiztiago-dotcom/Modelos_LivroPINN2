#!/usr/bin/env python3
"""
LSV + Gyöngy — Mobilidade de portadores em canais sub-2 nm
Capítulo 21 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.lsv_dinamica import simular_lsv
from src.gyongy_calibracao import calibrar_L_gyongy
from src.fator_local import mobilidade_efetiva_dupire, velocidade_saturacao
from src.rede_pinn_gyongy import RedePINN1D
from src.treinamento_gyongy import treinar_gyongy


def principal():
    print("=" * 70)
    print("  LSV + Gyöngy · Mobilidade em canal sub-2 nm")
    print("  Capítulo 21 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Simulando dinâmica LSV (Dupire–Heston adaptada)...")
    traj = simular_lsv(
        n_passos=3000, dt=0.005, E0=0.8, nu0=1.0,
        kappa=2.0, theta=1.0, xi=0.45, mu0=1.0, E_sat=1.5, semente=42,
    )
    print(f"    E: média={traj['E'].mean():.3f}, ν: média={traj['nu'].mean():.3f}")

    grade_E = np.linspace(0.05, 3.0, 60)
    _, E_nu, L_emp = calibrar_L_gyongy(traj["E"], traj["nu"], grade_E)
    mu = mobilidade_efetiva_dupire(grade_E)
    print(f"\n[2] Calibração empírica Gyöngy: L(E) em {len(grade_E)} pontos")

    print("\n[3] PINN para L(E) sob condição de Gyöngy...")
    rede = RedePINN1D(camadas=[1, 24, 24, 1], semente=42)
    res = treinar_gyongy(rede, grade_E, E_nu, n_epocas=350, taxa=1e-3, verbose_cada=50)
    L_pinn = rede.prever(grade_E)
    print(f"    Perda final: {res['perda_final']:.4e}")

    v_sat = velocidade_saturacao(grade_E)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))
    t = traj["t"]

    ax = eixos[0, 0]
    ax.plot(t, traj["E"], "C0-", lw=0.8, alpha=0.8, label=r"$E_t$")
    ax.plot(t, traj["nu"], "C3-", lw=0.8, alpha=0.7, label=r"$\nu_t$ (CIR)")
    ax.set_xlabel(r"$t$")
    ax.set_title("(a) Trajetória LSV (campo e variância fonônica)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(grade_E, mu, "k-", lw=2, label=r"$\mu_{\mathrm{eff}}(E)$")
    ax.plot(grade_E, v_sat, "C1--", lw=1.5, label=r"$v(E)/v_{\mathrm{sat}}$")
    ax.set_xlabel(r"$E$ (campo)")
    ax.set_title("(b) Mobilidade e saturação de velocidade")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(grade_E, L_emp, "C0-", lw=2, label=r"$L(E)$ empírico")
    ax.plot(grade_E, L_pinn, "C2--", lw=2, label=r"$L_\theta(E)$ PINN")
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$L(E)$")
    ax.set_title("(c) Fator local (condição de Gyöngy)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época")
    ax.set_ylabel("perda Gyöngy")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "lsv_gyongy_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação LSV–Gyöngy concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
