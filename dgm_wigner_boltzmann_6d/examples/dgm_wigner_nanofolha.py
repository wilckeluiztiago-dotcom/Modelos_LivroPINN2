#!/usr/bin/env python3
"""
DGM para equação de Wigner–Boltzmann em nanofolha ~1.6 nm
Transporte quase-balístico no espaço de fases.

Autor: Luiz Tiago Wilcke
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.wigner_boltzmann import NanofolhaWigner
from src.celula_dgm import RedeDGM
from src.treinamento import treinar_dgm
from src.utils import amostragem_lhs
from src.residuo_wigner import residuo_wigner_reduzido


def condicao_inicial(x: np.ndarray, kx: np.ndarray) -> np.ndarray:
    """Pacote gaussiano em (x, kx) — injeção quase-balística."""
    return np.exp(-((x - 0.2) / 0.12) ** 2 - ((kx - 1.5) / 0.8) ** 2)


def principal():
    print("=" * 70)
    print("  DGM — Wigner–Boltzmann em Nanofolha 1.6 nm")
    print("  Espaço de fases (redução operacional x, kx, t)")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    nano = NanofolhaWigner(Lx=1.0, Lz=0.16, V0=0.20, gamma_scatt=0.04)
    limites = nano.limites_fase_reduzida()
    print(f"\n[1] Nanofolha Lz={nano.Lz*10:.1f} nm (unidade 10 nm)")
    print(f"    Barreira V0={nano.V0}, γ={nano.gamma_scatt}")

    n_col = 300
    X_col = amostragem_lhs(n_col, limites, semente=42)
    print(f"    Pontos de colocation (LHS): {n_col}")

    # condição inicial t=0
    n0 = 100
    lim0 = limites.copy()
    lim0[2] = [0.0, 0.0]
    X0 = amostragem_lhs(n0, lim0, semente=1)
    X0[:, 2] = 0.0
    f0 = condicao_inicial(X0[:, 0], X0[:, 1])

    print("\n[2] Rede DGM (células recorrentes)...")
    rede = RedeDGM(dim_entrada=3, dim_oculta=48, n_camadas=2, semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")

    print("\n[3] Treinamento DGM...")
    res = treinar_dgm(
        rede, X_col, X0, f0, nano,
        n_epocas=350,
        taxa=6e-4,
        peso_pde=1.0,
        peso_ic=15.0,
        semente=0,
        verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # Avaliar residual
    r = residuo_wigner_reduzido(rede, X_col, nano)
    print(f"    |resíduo| médio: {np.mean(np.abs(r)):.4e}")

    # Mapas f_W(x, kx) em t fixo
    print("\n[4] Mapas no espaço de fases...")
    nx, nk = 50, 40
    xg = np.linspace(0, nano.Lx, nx)
    kg = np.linspace(-3.5, 3.5, nk)
    XX, KK = np.meshgrid(xg, kg)

    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    for ax, tval, titulo in zip(
        [eixos[0, 0], eixos[0, 1], eixos[1, 0]],
        [0.0, 0.4, 0.8],
        ["(a) t = 0", "(b) t = 0.4", "(c) t = 0.8"],
    ):
        pts = np.column_stack([XX.ravel(), KK.ravel(), np.full(XX.size, tval)])
        F = rede.prever(pts).reshape(XX.shape)
        cf = ax.contourf(XX, KK, F, levels=20, cmap="magma")
        plt.colorbar(cf, ax=ax, fraction=0.046)
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$k_x$")
        ax.set_title(titulo)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], color="C0", lw=1.5)
    ax.set_xlabel("época")
    ax.set_ylabel("perda DGM")
    ax.set_title("(d) Histórico de treinamento")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "dgm_wigner_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    # potencial
    fig2, ax2 = plt.subplots(figsize=(7, 3.5))
    xx = np.linspace(0, nano.Lx, 200)
    ax2.plot(xx, nano.potencial_efetivo(xx), "k-", lw=2)
    ax2.set_xlabel(r"$x$")
    ax2.set_ylabel(r"$V(x)$")
    ax2.set_title("Potencial efetivo na nanofolha (quase-balístico)")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "potencial_nanofolha.png"), dpi=140)

    print("\n" + "=" * 70)
    print("  Simulação DGM–Wigner concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
