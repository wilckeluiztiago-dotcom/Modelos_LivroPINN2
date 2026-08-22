#!/usr/bin/env python3
"""
DBRTD · Dupla Barreira · Tsu–Esaki · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_dbrtd import (
    parametros_dbrtd_default, potencial_dupla_barreira,
    transmissao_transfer_matrix, curva_JV,
)
from src.rede_pinn_dbrtd import RedePINN_DBRTD
from src.treinamento_dbrtd import treinar_dbrtd


def principal():
    print("=" * 70)
    print("  DBRTD · Tsu–Esaki · NDR · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_dbrtd_default()
    print(f"\n[1] Device: {dev}, V0={p['V0']}, E_F={p['E_F']}")

    x = np.linspace(0, p["L_total"], 300)
    V0 = potencial_dupla_barreira(x, 0.0, p)

    # transmissão vs E
    E_vals = np.linspace(0.1, 6.0, 80)
    T0 = np.array([transmissao_transfer_matrix(e, 0.0, p) for e in E_vals])
    print(f"    T max (V=0) = {T0.max():.4f} em E≈{E_vals[np.argmax(T0)]:.3f}")

    # curva J-V
    print("\n[2] Curva J–V (Tsu–Esaki)...")
    V_bias = np.linspace(0.05, 4.0, 35)
    J = curva_JV(V_bias, p)
    # NDR: dJ/dV < 0
    dJ = np.gradient(J, V_bias)
    if np.any(dJ < 0):
        print(f"    NDR detectado: min dJ/dV = {dJ.min():.4f}")
    else:
        print(f"    J max = {J.max():.4f}")

    # PINN em uma energia próxima à ressonância
    E_res = float(E_vals[np.argmax(T0)])
    print(f"\n[3] Treinando PINN ψ(x,E={E_res:.2f})...")
    n_col = 300
    x_c = torch.rand(n_col, 1, device=dev) * p["L_total"]
    E_c = torch.full((n_col, 1), E_res, device=dev)
    xE_col = torch.cat([x_c, E_c], dim=1).requires_grad_(True)

    n_bc = 20
    E_bc = torch.full((n_bc, 1), E_res, device=dev)
    xL = torch.zeros(n_bc, 1, device=dev)
    xR = torch.full((n_bc, 1), p["L_total"], device=dev)
    xE_L = torch.cat([xL, E_bc], 1).requires_grad_(True)
    xE_R = torch.cat([xR, E_bc], 1).requires_grad_(True)

    rede = RedePINN_DBRTD([2, 48, 48, 48, 2]).to(dev)
    res = treinar_dbrtd(
        rede, xE_col, xE_L, xE_R, V_bias=0.0, p=p,
        n_epocas=2000, taxa=1e-3, verbose_cada=200,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    xg = torch.linspace(0, p["L_total"], 200, device=dev).reshape(-1, 1)
    Eg = torch.full_like(xg, E_res)
    xEg = torch.cat([xg, Eg], 1)
    with torch.no_grad():
        psiR, psiI = rede.psi(xEg)
    dens = (psiR ** 2 + psiI ** 2).cpu().numpy().ravel()
    x_np = xg.cpu().numpy().ravel()

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x, V0, "k-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$V(x)$")
    ax.set_title("(a) Potencial de dupla barreira")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.semilogy(E_vals, np.maximum(T0, 1e-8), "C0-", lw=2)
    ax.axvline(E_res, color="C3", ls="--", label=fr"$E_{{\mathrm{{res}}}}={E_res:.2f}$")
    ax.set_xlabel(r"$E$"); ax.set_ylabel(r"$\mathcal{T}(E)$")
    ax.set_title("(b) Transmissão ressonante")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(V_bias, J, "C3-", lw=2)
    ax.set_xlabel(r"$V$"); ax.set_ylabel(r"$J(V)$")
    ax.set_title("(c) Curva J–V (Tsu–Esaki) / NDR")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.plot(x_np, dens, "C2-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$|\psi|^2$")
    ax.set_title(r"(d) PINN $|\psi(x)|^2$ na ressonância")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "dbrtd_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  DBRTD Tsu–Esaki concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
