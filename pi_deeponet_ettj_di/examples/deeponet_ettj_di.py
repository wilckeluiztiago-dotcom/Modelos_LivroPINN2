#!/usr/bin/env python3
"""
PI-DeepONet · ETTJ DI Futuro B3 (252 DU, capitalização composta)
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.curva_di_b3 import (
    gerar_curva_di, gerar_superficie_P, VERTICES_DI,
    forward_instantanea, preco_titulo_df,
)
from src.rede_deeponet import PIDeepONet
from src.treinamento_deeponet import treinar_deeponet


def principal():
    print("=" * 70)
    print("  PI-DeepONet · Curva DI Futuro B3 (ETTJ)")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    print("\n[1] Curva DI sintética (vértices líquidos)...")
    vertices, taxas = gerar_curva_di(r0=0.1215, slope=0.008, curvatura=-0.0015, semente=42)
    nomes = ["DI1N25", "DI1F26", "DI1F27", "DI1F28", "DI1F29", "DI1F31", "DI1F34", "DI1F36"]
    for n, v, r in zip(nomes, vertices, taxas):
        print(f"    {n}: τ={v:.2f}a  r={r*100:.2f}% a.a.")

    t_grid = np.linspace(0, 5, 25)
    T_grid = np.linspace(0.25, 10, 40)
    P_true = gerar_superficie_P(vertices, taxas, t_grid, T_grid)

    # dados de treino
    g = np.random.default_rng(0)
    n_dados = 200
    ti = g.uniform(0, 4, n_dados)
    Ti = g.uniform(0.25, 10, n_dados)
    Ti = np.maximum(Ti, ti + 0.1)
    tT_dados = np.column_stack([ti, Ti])
    P_dados = np.array([preco_titulo_df(vertices, taxas, t, T) for t, T in tT_dados])

    tT_col = np.column_stack([
        g.uniform(0, 4, 150),
        g.uniform(0.5, 10, 150),
    ])

    print("\n[2] PI-DeepONet (Branch=curva DI, Trunk=(t,T))...")
    rede = PIDeepONet(n_vertices=len(vertices), dim=24, semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_deeponet(
        rede, taxas, tT_dados, P_dados, tT_col,
        r_curto=float(taxas[0]),
        n_epocas=350, taxa=1e-3, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # superfície predita
    print("\n[3] Superfície P(t,T) predita...")
    TT, tt = np.meshgrid(T_grid, t_grid)
    pts = np.column_stack([tt.ravel(), TT.ravel()])
    P_pred = rede.prever(taxas, pts).reshape(tt.shape)

    # curva de taxas implícitas em t=0
    P0 = rede.prever(taxas, np.column_stack([np.zeros_like(T_grid), T_grid]))
    # r implícita: P = 1/(1+r)^T → r = P^(-1/T) - 1
    r_impl = np.power(np.maximum(P0, 1e-6), -1.0 / np.maximum(T_grid, 0.1)) - 1.0

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(vertices, taxas * 100, "ko-", lw=2, ms=8, label="vértices DI")
    ax.plot(T_grid, r_impl * 100, "C0--", lw=2, label="DeepONet implícita")
    ax.set_xlabel(r"$\tau$ (anos)")
    ax.set_ylabel(r"taxa (% a.a.)")
    ax.set_title("(a) ETTJ DI — vértices B3 e interpolação")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    im = ax.contourf(T_grid, t_grid, P_true, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(b) $P(t,T)$ referência")

    ax = eixos[1, 0]
    im = ax.contourf(T_grid, t_grid, P_pred, levels=20, cmap="viridis")
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xlabel(r"$T$"); ax.set_ylabel(r"$t$")
    ax.set_title(r"(c) $P_\theta(t,T)$ PI-DeepONet")

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "deeponet_ettj_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  PI-DeepONet ETTJ DI concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
