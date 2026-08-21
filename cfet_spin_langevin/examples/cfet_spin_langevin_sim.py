#!/usr/bin/env python3
"""
Acoplamento Espin–Langevin para CFET quântico (nFET sobre pFET)

Autor: Luiz Tiago Wilcke — Apêndice J.3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.acoplamento_cfet import CFETSpinLangevin
from src.langevin_potencial import potencial_nao_linear


def principal():
    print("=" * 70)
    print("  Espin–Langevin · CFET Quântico (nFET / pFET empilhados)")
    print("  Apêndice J.3 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    cfet = CFETSpinLangevin(
        n_spins=24,
        J_intra=1.0,
        J_inter=0.40,
        kappa_ep=0.45,
        beta=1.2,
        sigma=0.12,
        semente=42,
    )
    print("\n[1] CFET: nFET sobre pFET, barreira sub-nm")
    print(f"    J_inter={cfet.J_inter}, κ_ep={cfet.kappa_ep}")

    print("\n[2] Simulando dinâmica acoplada Glauber + Langevin...")
    hist = cfet.simular(n_passos=2500, dt=0.01)
    print(f"    m_n final={hist['m_n'][-1]:.3f}, m_p final={hist['m_p'][-1]:.3f}")
    print(f"    φ_n final={hist['phi_n'][-1]:.3f}, φ_p final={hist['phi_p'][-1]:.3f}")

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))
    t = hist["t"]

    ax = eixos[0, 0]
    ax.plot(t, hist["m_n"], "C0-", lw=1.2, label=r"$m_n$ (nFET)")
    ax.plot(t, hist["m_p"], "C3-", lw=1.2, label=r"$m_p$ (pFET)")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("magnetização / ocupação")
    ax.set_title("(a) Dinâmica de Glauber–Ising (sub-bandas)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(t, hist["phi_n"], "C0-", lw=1.2, label=r"$\phi_n$")
    ax.plot(t, hist["phi_p"], "C3-", lw=1.2, label=r"$\phi_p$")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"potencial de canal $\phi$")
    ax.set_title("(b) Langevin acoplado (potencial contínuo)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.scatter(hist["m_n"][::5], hist["phi_n"][::5], s=6, c="C0", alpha=0.5, label="nFET")
    ax.scatter(hist["m_p"][::5], hist["phi_p"][::5], s=6, c="C3", alpha=0.5, label="pFET")
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(r"$\phi$")
    ax.set_title("(c) Espaço de fases ocupação–potencial")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    phi = np.linspace(-2, 2, 200)
    ax.plot(phi, potencial_nao_linear(phi, a=1.0, b=0.25, c=0.0), "k-", lw=2)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$U(\phi)$")
    ax.set_title("(d) Potencial não-linear de Langevin")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "cfet_spin_langevin_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    # esquema CFET
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.add_patch(plt.Rectangle((0.1, 0.55), 0.8, 0.30, facecolor="#42A5F5", edgecolor="k", lw=2))
    ax2.add_patch(plt.Rectangle((0.1, 0.15), 0.8, 0.30, facecolor="#EF5350", edgecolor="k", lw=2))
    ax2.add_patch(plt.Rectangle((0.1, 0.45), 0.8, 0.10, facecolor="#FFECB3", edgecolor="k", lw=1.5))
    ax2.text(0.5, 0.70, "nFET  ·  Ising $m_n$ + Langevin $\\phi_n$", ha="center", va="center", fontsize=11, fontweight="bold")
    ax2.text(0.5, 0.30, "pFET  ·  Ising $m_p$ + Langevin $\\phi_p$", ha="center", va="center", fontsize=11, fontweight="bold")
    ax2.text(0.5, 0.50, "barreira dielétrica sub-nm  ·  $J_{inter}$, $\\kappa_{ep}$", ha="center", va="center", fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis("off")
    ax2.set_title("CFET: canais empilhados verticalmente", fontsize=12)
    fig2.savefig(os.path.join(os.path.dirname(caminho), "cfet_esquema.png"), dpi=140, bbox_inches="tight")

    print("\n" + "=" * 70)
    print("  Simulação CFET Espin–Langevin concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
