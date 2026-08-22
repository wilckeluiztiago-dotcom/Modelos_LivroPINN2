#!/usr/bin/env python3
"""
Tight-Binding sp³d⁵s* · Doador P em Si · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_tb import (
    parametros_tb_default, gerar_cluster_diamante, montar_hamiltoniano,
    diagonalizar_tb, ORBITAIS, N_ORB, V_P_screened,
)
from src.rede_pinn_tb import RedePINN_TB
from src.treinamento_tb import treinar_tb


def principal():
    print("=" * 70)
    print("  Tight-Binding sp³d⁵s* · P:Si · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_tb_default()
    print(f"\n[1] Device: {dev}")

    pos, idx_P = gerar_cluster_diamante(n_shells=1, a=p["a"])
    print(f"    Átomos no cluster: {len(pos)}, dim H = {len(pos)*N_ORB}")

    H = montar_hamiltoniano(pos, idx_P, p)
    evals, evecs = diagonalizar_tb(H, n_estados=4)
    print(f"    E_TB (4 menores): {evals.round(4)}")
    C0 = evecs[:, 0]
    # normalizar
    C0 = C0 / np.linalg.norm(C0)

    print("\n[2] Treinando PINN C_α(r)...")
    rede = RedePINN_TB(n_orb=N_ORB, camadas=[3, 48, 48, N_ORB]).to(dev)
    with torch.no_grad():
        rede.log_neg_E.copy_(torch.log(torch.tensor(max(-evals[0], 0.1), device=dev)))
    res = treinar_tb(rede, H, pos, C_ref=C0, n_epocas=2500, taxa=1e-3, verbose_cada=250)
    print(f"    Perda final: {res['perda_final']:.4e}")
    print(f"    E_PINN = {float(rede.energia().detach()):.4f}")

    # densidade por sítio |ψ|²
    dens_tb = np.array([np.sum(C0[i*N_ORB:(i+1)*N_ORB]**2) for i in range(len(pos))])
    r_t = torch.tensor(pos, dtype=torch.float32, device=dev)
    with torch.no_grad():
        C_nn = rede(r_t).cpu().numpy()
    dens_nn = np.sum(C_nn ** 2, axis=1)
    dens_nn = dens_nn / dens_nn.sum() * dens_tb.sum()

    # V_P radial
    rs = np.linspace(0.05, 4, 100)
    Vp = [V_P_screened(np.array([r, 0, 0]), eps_r=p["eps_r"], r_core=p["r_core"], U_cc=0.0) for r in rs]

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.bar(np.arange(len(pos)), dens_tb, alpha=0.7, label="TB exact", color="C0")
    ax.plot(np.arange(len(pos)), dens_nn, "C3o-", ms=4, label="PINN")
    ax.set_xlabel("índice do átomo"); ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title("(a) Densidade por sítio (estado fundamental)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(rs, Vp, "k-", lw=2)
    ax.set_xlabel(r"$r$"); ax.set_ylabel(r"$V_P(r)$")
    ax.set_title(r"(b) Potencial de Coulomb triado")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    # pesos orbitais no P (átomo 0)
    w_tb = C0[0:N_ORB] ** 2
    w_nn = C_nn[0] ** 2
    x = np.arange(N_ORB)
    ax.bar(x - 0.15, w_tb, 0.3, label="TB", color="C0")
    ax.bar(x + 0.15, w_nn / max(w_nn.sum(), 1e-12) * w_tb.sum(), 0.3, label="PINN", color="C3")
    ax.set_xticks(x); ax.set_xticklabels(ORBITAIS, rotation=45, fontsize=7)
    ax.set_ylabel(r"$|C_\alpha|^2$ no P")
    ax.set_title(r"(c) Composição orbital em $^{31}$P")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "tb_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  TB sp³d⁵s* concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
