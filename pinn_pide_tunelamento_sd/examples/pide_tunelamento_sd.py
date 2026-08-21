#!/usr/bin/env python3
"""
PINN Integro-Diferencial (PIDE) — Tunelamento Source–Drain sub-12 nm

Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.barreira_tunelamento import CanalSub12nm
from src.rede_pinn import RedePINN
from src.treinamento import treinar_pide
from src.residuo_pide import corrente_drift_diffusion, operador_tunelamento_mc
from src.utils import amostragem_lhs_1d


def principal():
    print("=" * 70)
    print("  PINN–PIDE · Tunelamento Source–Drain sub-12 nm")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    canal = CanalSub12nm(L=1.0, V_barreira=0.35, V_drain=0.25)
    print(f"\n[1] Canal L≈10 nm (sub-12 nm), V_barreira={canal.V_barreira}")

    # transmissão WKB vs energia
    Es = np.linspace(0.0, 0.5, 30)
    Ts = [canal.transmissao_wkb(E) for E in Es]
    print(f"    T_WKB(E=0.1) ≈ {canal.transmissao_wkb(0.1):.3e}")

    n_col = 80
    x_col = amostragem_lhs_1d(n_col, 0.02, 0.98, semente=42)
    x_bc = np.array([0.0, 1.0])
    n_bc = np.array([1.0, 0.15])  # injeção source, drenagem drain

    print("\n[2] Rede PINN [1,32,32,16,1]...")
    rede = RedePINN(camadas=[1, 32, 32, 16, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")

    print("\n[3] Treinamento PIDE (MC contínuo + continuidade)...")
    res = treinar_pide(
        rede, x_col, x_bc, n_bc, canal,
        n_epocas=300,
        taxa=6e-4,
        peso_pde=1.0,
        peso_bc=18.0,
        n_mc=12,
        semente=0,
        verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    x = np.linspace(0, 1, 120)
    n = rede.prever(x)
    n = np.maximum(n, 0.0)
    J = corrente_drift_diffusion(rede, x)
    G = operador_tunelamento_mc(rede, x, canal, n_mc=24, semente=7)
    V = canal.potencial(x)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x, V, "k-", lw=2)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$V(x)$")
    ax.set_title("(a) Barreira source–drain (sub-12 nm)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.semilogy(Es, np.maximum(Ts, 1e-16), "C3-", lw=2)
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$T_{\mathrm{WKB}}(E)$")
    ax.set_title("(b) Transmissão WKB da barreira")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(x, n, "C0-", lw=2, label=r"$n(x)$ PINN")
    ax.plot(x, G * 5, "C2--", lw=1.5, label=r"$G_{\mathrm{tun}}$ (×5)")
    ax.set_xlabel(r"$x$")
    ax.set_title("(c) Densidade e fonte de tunelamento")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época")
    ax.set_ylabel("perda PIDE")
    ax.set_title("(d) Treinamento")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "pide_tunelamento_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação PINN–PIDE concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
