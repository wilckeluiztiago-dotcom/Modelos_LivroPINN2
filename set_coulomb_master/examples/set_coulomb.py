#!/usr/bin/env python3
"""
Bloqueio de Coulomb · SET · Equação Mestre + PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.fisica_set import energia_carregamento, energia_livre, taxas_tunelamento
from src.equacao_mestre import simular_mestre, varredura_gate
from src.rede_pinn_mestre import RedePINN_Mestre
from src.treinamento_mestre import treinar_mestre


def principal():
    print("=" * 70)
    print("  Bloqueio de Coulomb · SET · Equação Mestre")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    C_Sigma, T = 1.0, 0.04
    E_c = energia_carregamento(C_Sigma)
    print(f"\n[1] E_c = {E_c:.3f}, kT = {T:.3f}  →  E_c/kT = {E_c/T:.1f} ≫ 1 (bloqueio)")

    print("\n[2] Evolução temporal P(N,t) via equação mestre...")
    traj = simular_mestre(
        N_min=-1, N_max=4, n_passos=400, dt=0.02,
        V_g=1.5, V_sd=0.3, T=T, C_Sigma=C_Sigma, semente=42,
    )
    print(f"    N: {traj['N_vals']}, P final={traj['P'][-1].round(3)}")

    print("\n[3] Varredura de gate → degraus de Coulomb...")
    Vg = np.linspace(0.0, 6.0, 80)
    var = varredura_gate(Vg, N_min=-1, N_max=4, V_sd=0.25, T=T, C_Sigma=C_Sigma, n_relax=300)
    print(f"    I max={var['I'].max():.4f}")

    print("\n[4] PINN P(N,t)...")
    N_vals = traj["N_vals"].astype(float)
    P0 = traj["P"][0]
    t_col = np.linspace(0.05, 2.0, 12)
    rede = RedePINN_Mestre(camadas=[2, 28, 28, 1], semente=42)
    print(f"    Parâmetros: {rede.n_parametros()}")
    res = treinar_mestre(
        rede, N_vals, t_col, P0,
        V_g=1.5, V_sd=0.3, T=T, C_Sigma=C_Sigma,
        n_epocas=250, taxa=8e-4, semente=0, verbose_cada=50,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    P_pinn = rede.prever_normalizado(N_vals, 2.0)

    print("\n[5] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    for i, N in enumerate(traj["N_vals"]):
        ax.plot(traj["t"], traj["P"][:, i], lw=1.5, label=f"N={N}")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$P(N,t)$")
    ax.set_title("(a) Evolução equação mestre")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(var["V_g"], var["I"], "C3-", lw=2)
    ax.set_xlabel(r"$V_g$"); ax.set_ylabel(r"$I$ (u.a.)")
    ax.set_title("(b) Degraus de Coulomb (I–V_g)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    width = 0.35
    ax.bar(N_vals - width / 2, traj["P"][-1], width, label="mestre", color="C0", alpha=0.8)
    ax.bar(N_vals + width / 2, P_pinn, width, label="PINN", color="C3", alpha=0.8)
    ax.set_xlabel(r"$N$"); ax.set_ylabel(r"$P(N)$")
    ax.set_title(r"(c) $P(N)$ estacionário: mestre vs PINN")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(np.maximum(res["historico"], 1e-12), "C4-", lw=1.5)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "set_coulomb_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  SET / Bloqueio de Coulomb concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
