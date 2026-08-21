#!/usr/bin/env python3
"""
B-PINNs · Incerteza epistêmica · RDF · Canal 1.6 nm
Apêndice C.3 — Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.rdf_dopantes import CanalRDF
from src.ensemble_bpinn import EnsembleBPINN


def principal():
    print("=" * 70)
    print("  B-PINN (ensemble) · RDF · Canal 1.6 nm · Apêndice C.3")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    canal = CanalRDF(L=1.0, n_dopantes=8, carga_dopante=1.2, largura_gauss=0.04, semente=42)
    print(f"\n[1] Canal ~1.6 nm, {canal.n_dopantes} dopantes")
    print(f"    Posições: {np.array2string(canal.posicoes, precision=3)}")

    x = np.linspace(0, 1, 100)
    rho = canal.densidade_carga(x)
    phi_ref = canal.potencial_referencia(x)

    g = np.random.default_rng(0)
    x_col = np.sort(g.uniform(0.05, 0.95, 50))
    rho_col = canal.densidade_carga(x_col)
    x_bc = np.array([0.0, 1.0])
    v_bc = np.array([canal.V_source, canal.V_drain])

    print("\n[2] Ensemble B-PINN (5 membros, posterior preditivo)...")
    ens = EnsembleBPINN(n_membros=5, camadas=[1, 24, 24, 1], semente=42)
    ens.treinar(x_col, rho_col, x_bc, v_bc, n_epocas=200, verbose=True)

    media, var = ens.prever_media_var(x)
    std = np.sqrt(np.maximum(var, 0.0))
    ic_lo, ic_hi = media - 1.96 * std, media + 1.96 * std

    print("\n[3] Ensemble de realizações RDF...")
    phis = np.stack([
        canal.nova_realizacao(semente=100 + s).potencial_referencia(x)
        for s in range(25)
    ], axis=0)
    phi_rdf_std = phis.std(axis=0)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x, rho, "k-", lw=1.5)
    for p in canal.posicoes:
        ax.axvline(p, color="C3", ls="--", alpha=0.5)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\rho$")
    ax.set_title("(a) Carga RDF (dopantes)"); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(x, phi_ref, "k-", lw=2, label="Poisson ref.")
    ax.plot(x, media, "C0-", lw=2, label=r"$\mathbb{E}[\phi]$ B-PINN")
    ax.fill_between(x, ic_lo, ic_hi, color="C0", alpha=0.25, label="IC 95%")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi$")
    ax.set_title("(b) Potencial + incerteza epistêmica")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(x, std, "C2-", lw=2, label=r"std$_\theta$ (B-PINN)")
    ax.plot(x, phi_rdf_std, "C3--", lw=1.5, label=r"std$_{\mathrm{RDF}}$")
    ax.set_xlabel(r"$x$"); ax.set_ylabel("desvio padrão")
    ax.set_title("(c) Mapa de variância"); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    for h in ens.historicos:
        ax.semilogy(h, alpha=0.7, lw=1)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treino dos membros"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "bpinn_rdf_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Simulação B-PINN RDF concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
