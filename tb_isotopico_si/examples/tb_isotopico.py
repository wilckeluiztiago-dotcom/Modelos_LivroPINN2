#!/usr/bin/env python3
"""
TB com Flutuação Isotópica Si²⁸/²⁹/³⁰ · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_iso import (
    parametros_iso_default, amostrar_massas, delta_epsilon,
    hamiltoniano_iso, espectro_ensemble, splitting_T2_like, M_BAR, MASSAS,
)
from src.rede_pinn_iso import RedePINN_Iso
from src.treinamento_iso import treinar_iso


def principal():
    print("=" * 70)
    print("  TB Isotópico · Si²⁸/Si²⁹/Si³⁰ · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_iso_default()
    print(f"\n[1] Device: {dev}, M̄={M_BAR:.4f} u, α_iso={p['alpha_iso']}")

    # referência sem desordem
    de0 = np.zeros(p["n_sites"])
    H0 = hamiltoniano_iso(p["n_sites"], de0, p["t_hop"], p["E0"], p["V_P"])
    e0 = np.linalg.eigvalsh(H0)[:p["n_estados"]]
    print(f"    E sem desordem: {e0.round(4)}")

    print("\n[2] Ensemble isotópico...")
    evals, Hs, deltas = espectro_ensemble(p, semente=42)
    split = splitting_T2_like(evals)
    print(f"    ⟨split T₂-like⟩ = {split.mean():.4e} ± {split.std():.4e}")

    # uma realização para PINN
    H = Hs[0]
    evals1, evecs1 = np.linalg.eigh(H)
    C_ref = evecs1[:, 0]
    C_ref = C_ref / np.linalg.norm(C_ref)

    print("\n[3] Treinando PINN (realização #0)...")
    rede = RedePINN_Iso(n_sites=p["n_sites"]).to(dev)
    with torch.no_grad():
        rede.log_neg_E.copy_(torch.log(torch.tensor(max(-evals1[0], 0.1), device=dev)))
    res = treinar_iso(rede, H, C_ref=C_ref, n_epocas=2000, taxa=1e-3, verbose_cada=200)
    print(f"    Perda final: {res['perda_final']:.4e}")
    print(f"    E_PINN = {float(rede.energia().detach()):.4f}  E_TB = {evals1[0]:.4f}")

    with torch.no_grad():
        C_nn = rede.vetor_C(dev).cpu().numpy()
    C_nn = C_nn / (np.linalg.norm(C_nn) + 1e-12)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.hist(deltas[0], bins=15, color="C0", alpha=0.8)
    ax.axvline(0, color="k", ls="--")
    ax.set_xlabel(r"$\delta\epsilon_i$"); ax.set_ylabel("sítios")
    ax.set_title(r"(a) Desordem isotópica $\delta\epsilon_i$ (1 realização)")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    for k in range(min(5, p["n_estados"])):
        ax.hist(evals[:, k], bins=20, alpha=0.5, label=fr"$E_{k}$")
    ax.set_xlabel(r"$E$"); ax.set_ylabel("contagem")
    ax.set_title("(b) Ensemble de autovalores")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    sites = np.arange(p["n_sites"])
    ax.plot(sites, C_ref ** 2, "k--", lw=2, label="TB exact")
    ax.plot(sites, C_nn ** 2, "C3-", lw=1.5, label="PINN")
    ax.set_xlabel("sítio"); ax.set_ylabel(r"$|C_i|^2$")
    ax.set_title("(c) Densidade do estado fundamental")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.hist(split, bins=20, color="C2", alpha=0.85)
    ax.axvline(split.mean(), color="k", ls="--", label=fr"média={split.mean():.3e}")
    ax.set_xlabel(r"$\Delta E_{T_2\mathrm{-like}}$"); ax.set_ylabel("contagem")
    ax.set_title(r"(d) Splitting residual (quebra de degenerescência)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "iso_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    # treino
    fig2, ax2 = plt.subplots(figsize=(6, 3.5))
    ax2.semilogy(res["historico"], "C4-", lw=1.2)
    ax2.set_xlabel("época"); ax2.set_ylabel("perda")
    ax2.set_title("Treinamento PINN")
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(os.path.dirname(caminho), "iso_treino.png"), dpi=120)

    print("\n" + "=" * 70)
    print("  TB isotópico concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
