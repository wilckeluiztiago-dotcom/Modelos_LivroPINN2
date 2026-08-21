#!/usr/bin/env python3
"""
Fokker–Planck + Landau · Chaveamento TFET / RTD 1.6 nm
Cap. 41 & Apêndice J.2 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.potencial_landau import potencial_landau, tempo_kramers, barreira_e_minimos, forca_landau
from src.langevin_fp import simular_langevin, densidade_estacionaria_analitica
from src.rede_pinn_fp import RedePINN_FP
from src.treinamento_fp import treinar_fp


def principal():
    print("=" * 70)
    print("  Fokker–Planck + Landau · TFET / RTD 1.6 nm")
    print("  Cap. 41 & Apêndice J.2 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    a, b, sigma = 1.0, 1.0, 0.45
    x_min, x_bar, dV = barreira_e_minimos(a, b)
    tau_K = tempo_kramers(a, b, sigma)
    print(f"\n[1] Potencial Landau: x_min=±{x_min:.3f}, ΔV={dV:.3f}")
    print(f"    Tempo de Kramers τ_K ≈ {tau_K:.3e}")

    print("\n[2] Simulando Langevin (escape entre poços)...")
    traj = simular_langevin(n_passos=8000, dt=0.01, x0=-1.0, a=a, b=b, sigma=sigma, semente=42)

    # densidade estacionária
    xg = np.linspace(-2.2, 2.2, 200)
    p_inf = densidade_estacionaria_analitica(xg, a, b, sigma)
    V = potencial_landau(xg, a, b)

    # PINN para p(x,t)
    print("\n[3] PINN Fokker–Planck p(x,t)...")
    g = np.random.default_rng(0)
    n_col = 400
    X_col = np.column_stack([
        g.uniform(-2.0, 2.0, n_col),
        g.uniform(0.0, 2.0, n_col),
    ])
    # IC: gaussiana no poço esquerdo
    n0 = 80
    x0 = g.uniform(-2.0, 2.0, n0)
    X0 = np.column_stack([x0, np.zeros(n0)])
    p0 = np.exp(-0.5 * ((x0 + 1.0) / 0.3) ** 2)
    p0 = p0 / (np.trapezoid(np.exp(-0.5 * ((xg + 1.0) / 0.3) ** 2), xg) + 1e-12)

    rede = RedePINN_FP(camadas=[2, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_fp(
        rede, X_col, X0, p0, a=a, b=b, sigma=sigma,
        n_epocas=300, taxa=7e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # p PINN em t grande vs p_∞
    pts_T = np.column_stack([xg, np.full_like(xg, 2.0)])
    p_pinn = rede.prever(pts_T)
    p_pinn = p_pinn / (np.trapezoid(p_pinn, xg) + 1e-12)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(xg, V, "k-", lw=2)
    ax.axvline(-x_min, color="C0", ls="--", alpha=0.7, label="mínimos")
    ax.axvline(x_min, color="C0", ls="--", alpha=0.7)
    ax.axvline(0, color="C3", ls=":", alpha=0.7, label="barreira")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$V(x)$")
    ax.set_title(r"(a) Potencial de Landau $V=-\frac{a}{2}x^2+\frac{b}{4}x^4$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj["t"], traj["x"], "C0-", lw=0.7, alpha=0.8)
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$X_t$")
    ax.set_title(f"(b) Langevin — chaveamento (τ_K≈{tau_K:.2f})")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(xg, p_inf, "k-", lw=2, label=r"$p_\infty$ analítica")
    ax.plot(xg, p_pinn, "C2--", lw=2, label=r"$p_\theta$ PINN $(t=2)$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$p$")
    ax.set_title("(c) Densidade nos dois poços (pico–vale)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda FP")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "fp_landau_tfet_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação Fokker–Planck / Landau TFET concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
