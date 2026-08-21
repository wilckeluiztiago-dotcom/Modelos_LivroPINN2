#!/usr/bin/env python3
"""
PINN 3D livre de malhas — Poisson em GAAFET
∇ · (ε ∇φ) = −ρ

Autor: Luiz Tiago Wilcke
Base: livro de PINNs (Caps. 2–3)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src.geometria_gaafet import GeometriaGAAFET
from src.rede_pinn3d import RedePINN3D
from src.treinamento import treinar_pinn3d
from src.utils import amostragem_lhs
from src.residuo_poisson import residuo_poisson


def principal():
    print("=" * 70)
    print("  PINN 3D livre de malhas — Poisson em GAAFET")
    print("  ∇ · (ε ∇φ) = −ρ")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    geo = GeometriaGAAFET(
        L=1.0, R_canal=0.25, R_ox=0.40,
        V_gate=0.5, V_source=0.0, V_drain=0.3,
    )
    print(f"\n[1] Geometria GAAFET: L={geo.L}, R_canal={geo.R_canal}, R_ox={geo.R_ox}")

    # Pontos de colocation (LHS — Cap. 3.5)
    n_col = 200
    limites = geo.limites_dominio()
    X_col = amostragem_lhs(n_col, limites, semente=42)
    # filtra fora do óxido (domínio físico)
    r = np.sqrt(X_col[:, 1] ** 2 + X_col[:, 2] ** 2)
    X_col = X_col[r <= geo.R_ox * 1.05]
    print(f"    Pontos de colocation (LHS): {len(X_col)}")

    # Contornos
    Xg, Vg = geo.pontos_contorno_gate(60, semente=1)
    Xs, Vs = geo.pontos_contorno_source_drain(40, semente=2)
    X_bc = np.vstack([Xg, Xs])
    valores_bc = np.concatenate([Vg, Vs])
    print(f"    Pontos de contorno: {len(X_bc)}")

    def epsilon_fn(x, y, z):
        return geo.permitividade(x, y, z)

    def rho_fn(x, y, z):
        return geo.densidade_carga(x, y, z, rho0=0.8)

    # Rede
    print("\n[2] Rede PINN 3D [3, 48, 48, 32, 1]...")
    rede = RedePINN3D(camadas=[3, 32, 32, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")

    print("\n[3] Treinamento (perda composta Cap. 2.5)...")
    res = treinar_pinn3d(
        rede, X_col, X_bc, valores_bc,
        epsilon_fn, rho_fn,
        n_epocas=300,
        taxa=6e-4,
        peso_pde=1.0,
        peso_bc=20.0,
        semente=0,
        verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # Avaliação em um plano longitudinal (z=0)
    print("\n[4] Mapeando potencial no plano z=0...")
    nx, ny = 40, 30
    xg = np.linspace(0, geo.L, nx)
    yg = np.linspace(-geo.R_ox, geo.R_ox, ny)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = np.zeros_like(XX)
    pts = np.column_stack([XX.ravel(), YY.ravel(), ZZ.ravel()])
    mask = np.sqrt(pts[:, 1] ** 2 + pts[:, 2] ** 2) <= geo.R_ox
    phi = np.full(len(pts), np.nan)
    phi[mask] = rede.prever(pts[mask])
    PHI = phi.reshape(XX.shape)

    res_col = residuo_poisson(rede, X_col, epsilon_fn, rho_fn)
    print(f"    |resíduo| médio (colocation): {np.mean(np.abs(res_col)):.4e}")

    # Figuras
    print("\n[5] Gerando figuras...")
    fig = plt.figure(figsize=(13, 10))

    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    # esboço do nanowire
    theta = np.linspace(0, 2 * np.pi, 40)
    for xi in np.linspace(0, geo.L, 6):
        ax1.plot(np.full_like(theta, xi), geo.R_canal * np.cos(theta),
                 geo.R_canal * np.sin(theta), "c-", alpha=0.5, lw=0.8)
        ax1.plot(np.full_like(theta, xi), geo.R_ox * np.cos(theta),
                 geo.R_ox * np.sin(theta), "m-", alpha=0.4, lw=0.8)
    ax1.scatter(X_col[::5, 0], X_col[::5, 1], X_col[::5, 2],
                c="lime", s=4, alpha=0.6, label="colocation")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_zlabel("z")
    ax1.set_title("(a) GAAFET + pontos LHS (livre de malha)")
    ax1.view_init(elev=18, azim=-55)

    ax2 = fig.add_subplot(2, 2, 2)
    cf = ax2.contourf(XX, YY, PHI, levels=20, cmap="viridis")
    plt.colorbar(cf, ax=ax2, label=r"$\phi$")
    circ1 = plt.Circle((0.5, 0), geo.R_canal, fill=False, color="w", ls="--", lw=1.2)
    ax2.add_patch(plt.Circle((geo.L / 2, 0), geo.R_canal, fill=False, color="w", ls="--"))
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title(r"(b) Potencial $\phi(x,y,z=0)$")
    ax2.set_aspect("equal")

    ax3 = fig.add_subplot(2, 2, 3)
    # corte radial no meio do canal
    y_line = np.linspace(-geo.R_ox, geo.R_ox, 80)
    pts_r = np.column_stack([
        np.full(80, geo.L / 2), y_line, np.zeros(80)
    ])
    phi_r = rede.prever(pts_r)
    ax3.plot(y_line, phi_r, "C0-", lw=2)
    ax3.axvline(-geo.R_canal, color="gray", ls="--", label="canal")
    ax3.axvline(geo.R_canal, color="gray", ls="--")
    ax3.axvline(-geo.R_ox, color="C3", ls=":", label="óxido/gate")
    ax3.axvline(geo.R_ox, color="C3", ls=":")
    ax3.set_xlabel("y (corte radial)")
    ax3.set_ylabel(r"$\phi$")
    ax3.set_title("(c) Perfil radial em x = L/2")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.semilogy(res["historico"], "C4-", lw=1.5)
    ax4.set_xlabel("época")
    ax4.set_ylabel("perda composta")
    ax4.set_title("(d) Treinamento (Cap. 2.5 / 3.6)")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "gaafet_pinn_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação GAAFET–PINN 3D concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
