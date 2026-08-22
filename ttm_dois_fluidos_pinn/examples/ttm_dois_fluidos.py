#!/usr/bin/env python3
"""
Modelo de Dois Fluidos (TTM) · Elétron–Fônon · PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.fisica_ttm import simular_ttm, parametros_ttm_default
from src.rede_pinn_ttm import RedePINN_TTM
from src.treinamento_ttm import treinar_ttm


def principal():
    print("=" * 70)
    print("  TTM · Dois Fluidos Elétron–Fônon · Nanotransistor 1 nm")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    p = parametros_ttm_default()
    print(f"\n[1] Parâmetros: C_e={p['C_e']}, C_L={p['C_L']}, G={p['G']}, κ_e={p['kappa_e']}")

    print("\n[2] Simulando TTM 1D (hotspot eletrônico)...")
    sim = simular_ttm(n_x=30, n_t=2000, L=1.0, t_final=1.0, Te0=1.0, TL0=1.0, hot_spot=0.25, p=p)
    print(f"    Te max final={sim['Te'][-1].max():.3f}, TL max={sim['TL'][-1].max():.3f}")

    print("\n[3] PINN (T_e, T_L)...")
    g = np.random.default_rng(0)
    n_col = 350
    X_col = np.column_stack([
        g.uniform(0, 1, n_col),
        g.uniform(0, 1.5, n_col),
    ])
    n0 = 50
    x0 = np.linspace(0, 1, n0)
    X0 = np.column_stack([x0, np.zeros(n0)])
    Te0 = 1.0 + 0.5 * np.exp(-((x0 - 0.5) / 0.15) ** 2)
    TL0 = np.ones(n0)

    rede = RedePINN_TTM(camadas=[2, 32, 32, 2], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_ttm(
        rede, X_col, X0, Te0, TL0, p,
        n_epocas=280, taxa=8e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfis no centro x=0.5
    tg = np.linspace(0, 1.5, 40)
    pts = np.column_stack([np.full_like(tg, 0.5), tg])
    out = rede.prever(pts)
    Te_nn, TL_nn = out[:, 0], out[:, 1]
    # referência da simulação no centro
    ic = len(sim["x"]) // 2
    Te_ref = sim["Te"][:, ic]
    TL_ref = sim["TL"][:, ic]

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    extent = [sim["x"][0], sim["x"][-1], sim["t"][0], sim["t"][-1]]
    im = ax.imshow(sim["Te"], aspect="auto", origin="lower", extent=extent, cmap="hot")
    plt.colorbar(im, ax=ax, fraction=0.046, label=r"$T_e$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(a) $T_e(x,t)$ — elétrons superaquecidos")

    ax = eixos[0, 1]
    im = ax.imshow(sim["TL"], aspect="auto", origin="lower", extent=extent, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, label=r"$T_L$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(b) $T_L(x,t)$ — rede cristalina")

    ax = eixos[1, 0]
    ax.plot(sim["t"], Te_ref, "C3-", lw=2, label=r"$T_e$ sim")
    ax.plot(sim["t"], TL_ref, "C0-", lw=2, label=r"$T_L$ sim")
    ax.plot(tg, Te_nn, "C3--", lw=1.5, label=r"$T_e$ PINN")
    ax.plot(tg, TL_nn, "C0--", lw=1.5, label=r"$T_L$ PINN")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$T$")
    ax.set_title(r"(c) Centro do canal: não-equilíbrio $T_e > T_L$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(np.maximum(res["historico"], 1e-12), "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN TTM")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "ttm_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  TTM dois fluidos concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
