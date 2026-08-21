#!/usr/bin/env python3
"""
DSGE Neo-Keynesiano · Regra de Taylor Copom/Bacen
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.modelo_nk import ParametrosNK, simular_dsge, impulso_resposta, solucao_estatica_nk
from src.rede_politica import RedePoliticaNK
from src.treinamento_politica import treinar_politica


def principal():
    print("=" * 70)
    print("  DSGE Neo-Keynesiano · Taylor Rule Copom/Bacen")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    p = ParametrosNK(phi_pi=1.5, phi_y=0.125, kappa=0.12, sigma=1.0)
    print(f"\n[1] Parâmetros: φ_π={p.phi_pi}, φ_y={p.phi_y}, κ={p.kappa}")

    print("\n[2] Simulando economia com choques (rn, TT, fiscal)...")
    sim = simular_dsge(n_periodos=60, p=p, semente=42)
    print(f"    ŷ médio={sim['y'].mean():.4f}, π̂ médio={sim['pi'].mean():.4f}")

    print("\n[3] Gerando dados de treino (solução RE)...")
    g = np.random.default_rng(0)
    n = 400
    rn = g.normal(0, 0.015, n)
    tt = g.normal(0, 0.01, n)
    fisc = g.normal(0, 0.006, n)
    estados = np.column_stack([rn, tt, fisc])
    alvos = np.array([
        solucao_estatica_nk(rn[j], tt[j], fisc[j], 0.0, p) for j in range(n)
    ])

    print("\n[4] Rede de política (ŷ, π̂, î)...")
    rede = RedePoliticaNK(camadas=[3, 32, 32, 3], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_politica(
        rede, estados, alvos, p,
        n_epocas=350, taxa=1e-3, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # IRFs
    irf_rn = impulso_resposta("rn", 0.01, 30, p)
    irf_tt = impulso_resposta("tt", 0.01, 30, p)

    # projeção NN vs RE
    pred = rede.prever(estados[:50])
    if pred.ndim == 1:
        pred = pred.reshape(1, -1)

    print("\n[5] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(sim["t"], sim["y"] * 100, "C0-", lw=1.2, label=r"$\hat y$")
    ax.plot(sim["t"], sim["pi"] * 100, "C3-", lw=1.2, label=r"$\hat\pi$")
    ax.plot(sim["t"], sim["i"] * 100, "C2-", lw=1.2, label=r"$\hat i$ (Selic)")
    ax.set_xlabel("trimestres"); ax.set_ylabel("% desvio")
    ax.set_title("(a) Simulação DSGE (choques rn/TT/fiscal)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(irf_rn["t"], irf_rn["y"] * 100, "C0-", lw=2, label=r"$\hat y$")
    ax.plot(irf_rn["t"], irf_rn["pi"] * 100, "C3-", lw=2, label=r"$\hat\pi$")
    ax.plot(irf_rn["t"], irf_rn["i"] * 100, "C2-", lw=2, label=r"$\hat i$")
    ax.set_xlabel("trimestres"); ax.set_title(r"(b) IRF choque $r^n$ (taxa natural)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(irf_tt["t"], irf_tt["y"] * 100, "C0-", lw=2, label=r"$\hat y$")
    ax.plot(irf_tt["t"], irf_tt["pi"] * 100, "C3-", lw=2, label=r"$\hat\pi$")
    ax.plot(irf_tt["t"], irf_tt["i"] * 100, "C2-", lw=2, label=r"$\hat i$")
    ax.set_xlabel("trimestres"); ax.set_title(r"(c) IRF choque termos de troca")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento rede de política")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "dsge_taylor_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  DSGE Taylor Bacen concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
