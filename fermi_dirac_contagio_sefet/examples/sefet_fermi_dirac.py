#!/usr/bin/env python3
"""
Contágio Fermi–Dirac · Single-Electron FET · Apêndice J.4
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.cadeia_dopantes import CadeiaDopantes
from src.contagio_fermi_dirac import simular_transporte, probabilidade_fermi_dirac
from src.pinn_condutancia import RedePINN, treinar_condutancia


def varredura_gate(
    V_gates: np.ndarray,
    n_sitios: int = 8,
    n_passos: int = 2000,
    beta: float = 1.2,
    U_coulomb: float = 0.4,
    semente: int = 42,
) -> np.ndarray:
    """Corrente média vs polarização de gate (oscilações de Coulomb)."""
    G = np.zeros_like(V_gates)
    for i, Vg in enumerate(V_gates):
        cadeia = CadeiaDopantes(
            n_sitios=n_sitios,
            U_coulomb=U_coulomb,
            desordem=0.12,
            V_source=0.0,
            V_drain=0.8,
            semente=semente + i,
        )
        cadeia.D0 = cadeia.D0 - 0.5 * Vg  # efeito de gate
        res = simular_transporte(
            cadeia, n_passos=n_passos, beta=beta,
            taxa_source=0.7, taxa_drain=0.7, semente=semente + i,
        )
        G[i] = res["corrente_media"]
    return G


def principal():
    print("=" * 70)
    print("  Contágio Fermi–Dirac · SEFET · Apêndice J.4")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    # curva P_ij
    print("\n[1] Probabilidades Fermi–Dirac P_ij...")
    dE = np.linspace(-3, 3, 100)
    P = [probabilidade_fermi_dirac(0.0, -e, beta=2.0) for e in dE]  # D_i=0, D_j=-e → Δ=e

    # trajetória única
    print("\n[2] Simulando transporte elétron-a-elétron...")
    cadeia = CadeiaDopantes(n_sitios=10, U_coulomb=0.6, desordem=0.25, semente=42)
    traj = simular_transporte(cadeia, n_passos=2500, beta=1.5, semente=42)
    print(f"    Corrente média: {traj['corrente_media']:.4f}")
    print(f"    ⟨N⟩ elétrons: {traj['n_eletrons'].mean():.3f}")

    # oscilações de Coulomb
    print("\n[3] Varredura de gate (oscilações de condutância)...")
    V_gates = np.linspace(-1.0, 2.0, 25)
    G_dados = varredura_gate(V_gates, n_sitios=8, n_passos=1200, beta=1.5, semente=7)
    print(f"    G_max={G_dados.max():.4f}, G_min={G_dados.min():.4f}")

    # PINN
    print("\n[4] PINN para mapa de condutância G(V_g)...")
    rede = RedePINN(camadas=[1, 32, 32, 1], semente=42)
    # normalizar V para rede
    V_norm = (V_gates - V_gates.mean()) / (V_gates.std() + 1e-8)
    res = treinar_condutancia(rede, V_norm, G_dados, n_epocas=350, taxa=1e-3, verbose_cada=50)
    G_pinn = rede.prever(V_norm)
    print(f"    Perda final: {res['perda_final']:.4e}")

    print("\n[5] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(dE, P, "k-", lw=2)
    ax.set_xlabel(r"$D_i - D_j$")
    ax.set_ylabel(r"$P_{ij}$")
    ax.set_title(r"(a) $P_{ij}=1/(1+e^{\beta(D_i-D_j)})$")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(traj["n_eletrons"], "C0-", lw=0.7, alpha=0.8)
    ax.set_xlabel("passo")
    ax.set_ylabel(r"$N(t)$")
    ax.set_title("(b) Número de elétrons (bloqueio de Coulomb)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(V_gates, G_dados, "C3o-", lw=1.5, ms=5, label="simulação")
    ax.plot(V_gates, G_pinn, "C0--", lw=2, label="PINN")
    ax.set_xlabel(r"$V_g$")
    ax.set_ylabel(r"$G$ (corrente média)")
    ax.set_title("(c) Oscilações de condutância (Coulomb)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.5)
    ax.set_xlabel("época")
    ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "sefet_fermi_dirac_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Simulação SEFET Fermi–Dirac concluída.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
