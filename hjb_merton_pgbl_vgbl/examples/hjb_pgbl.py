#!/usr/bin/env python3
"""
HJB-Merton · Portfólios PGBL/VGBL · Alocação e resgate ótimos
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.merton_crra import pi_otimo_merton, c_otimo_aprox, simular_riqueza, utilidade_crra
from src.rede_pinn_hjb import RedePINN_HJB
from src.treinamento_hjb import treinar_hjb


def principal():
    print("=" * 70)
    print("  HJB-Merton · PGBL/VGBL · Previc/Susep")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    mu, r, sigma, gamma, rho = 0.11, 0.085, 0.14, 2.0, 0.04
    pi_star = pi_otimo_merton(mu, r, sigma, gamma)
    print(f"\n[1] Política Merton: π*={pi_star:.2%} em multimercado")
    print(f"    CDI≈{r:.2%}, μ multimercado={mu:.2%}, γ={gamma}")

    print("\n[2] Simulando riqueza com aportes mensais...")
    traj = simular_riqueza(
        n_passos=360, x0=50.0, aporte=1.5,
        mu=mu, r=r, sigma=sigma, gamma=gamma, rho=rho, semente=42,
    )
    print(f"    Riqueza final: R$ {traj['x'][-1]:.1f} mil")

    print("\n[3] PINN HJB v(t,x)...")
    g = np.random.default_rng(0)
    n_col = 350
    X_col = np.column_stack([
        g.uniform(0.0, 1.0, n_col),      # t normalizado
        g.uniform(10.0, 500.0, n_col),   # riqueza
    ])
    # terminal: utilidade da riqueza residual
    n_term = 60
    xT = g.uniform(10, 500, n_term)
    X_term = np.column_stack([np.ones(n_term), xT])
    V_term = utilidade_crra(xT, gamma)

    rede = RedePINN_HJB(camadas=[2, 28, 28, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_hjb(
        rede, X_col, X_term, V_term,
        n_epocas=250, taxa=7e-4, semente=0, verbose_cada=50,
        mu=mu, r=r, sigma=sigma, gamma=gamma, rho=rho,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # políticas ao longo de x
    xg = np.linspace(20, 400, 50)
    c_star = np.array([c_otimo_aprox(x, gamma, rho, r, mu, sigma) for x in xg])
    pi_line = np.full_like(xg, pi_star)

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(traj["t"], traj["x"], "C0-", lw=1.5)
    ax.set_xlabel("anos"); ax.set_ylabel(r"riqueza $x_t$ (R\$ mil)")
    ax.set_title("(a) Trajetória PGBL/VGBL com aportes")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(xg, pi_line * 100, "C3-", lw=2, label=r"$\pi^*$ multimercado")
    ax.axhline((1 - pi_star) * 100, color="C0", ls="--", lw=1.5, label="renda fixa (CDI)")
    ax.set_xlabel(r"$x$"); ax.set_ylabel("% alocação")
    ax.set_title("(b) Alocação ótima RF vs multimercado")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(xg, c_star, "C2-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$c^*(x)$")
    ax.set_title("(c) Política de resgate / consumo ótimo")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(np.maximum(res["historico"], 1e-12), "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda HJB")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "hjb_pgbl_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  HJB-Merton PGBL/VGBL concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
