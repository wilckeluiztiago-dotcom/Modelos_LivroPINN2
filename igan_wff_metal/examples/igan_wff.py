#!/usr/bin/env python3
"""
I-GAN · Síntese de Work Function Fluctuation (WFF) · TiN/TaN 1.6 nm
Capítulo 44 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.graos_wff import gerar_mapa_graos, potencial_de_wf
from src.rede_gan import Gerador, Discriminador
from src.treinamento_igan import treinar_igan
from src.perda_igan import perda_fisica_poisson


def principal():
    print("=" * 70)
    print("  I-GAN · WFF metálico (TiN/TaN) · 1.6 nm")
    print("  Capítulo 44 — Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    nx, ny = 24, 12
    print(f"\n[1] Mapas de grãos reais {nx}×{ny}...")
    _, wf_real = gerar_mapa_graos(nx, ny, n_graos=10, semente=42)

    print("\n[2] Gerador + Discriminador...")
    G = Gerador(dim_z=12, nx=nx, ny=ny, semente=42)
    D = Discriminador(nx=nx, ny=ny, semente=43)
    print(f"    G params: {G.n_parametros()}, D params: {D.n_parametros()}")

    print("\n[3] Treinamento I-GAN (adversário + Poisson)...")
    hist = treinar_igan(
        G, D, n_epocas=150, batch=4,
        taxa_g=1e-3, taxa_d=1e-3, lambda_phys=0.5,
        semente=0, verbose_cada=50,
    )

    print("\n[4] Amostras sintéticas...")
    g = np.random.default_rng(99)
    z = g.normal(size=(4, G.dim_z))
    wf_fake = G.gerar(z)
    phys = perda_fisica_poisson(wf_fake)
    print(f"    Resíduo Poisson médio (fake): {phys:.4e}")

    print("\n[5] Figuras...")
    fig, eixos = plt.subplots(2, 3, figsize=(13, 8))

    ax = eixos[0, 0]
    im = ax.imshow(wf_real.T, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("(a) WF real (grãos TiN/TaN)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = eixos[0, 1]
    im = ax.imshow(wf_fake[0].T, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("(b) WF sintético G(z)")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = eixos[0, 2]
    im = ax.imshow(wf_fake[1].T, aspect="auto", cmap="coolwarm", origin="lower")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("(c) WF sintético G(z')")
    ax.set_xlabel("x"); ax.set_ylabel("y")

    ax = eixos[1, 0]
    phi_r = potencial_de_wf(wf_real)
    phi_f = potencial_de_wf(wf_fake[0])
    ax.plot(phi_r, "k-", lw=2, label=r"$\phi$ real")
    ax.plot(phi_f, "C1--", lw=2, label=r"$\phi_G$ sintético")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi$")
    ax.set_title("(d) Potencial eletrostático")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.plot(hist["hist_g"], "C0-", lw=1.2, label=r"$\mathcal{L}_G$")
    ax.plot(hist["hist_d"], "C3-", lw=1.2, label=r"$\mathcal{L}_D$")
    ax.set_xlabel("época"); ax.set_title("(e) Perdas adversárias")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 2]
    ax.semilogy(np.maximum(hist["hist_phys"], 1e-12), "C2-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel(r"$\|\nabla\cdot(\epsilon\nabla\phi_G)+\rho\|^2$")
    ax.set_title("(f) Perda física (Poisson)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "igan_wff_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Treinamento I-GAN WFF concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
