#!/usr/bin/env python3
"""
2CK Não-Fermi Líquido · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_2ck import (
    parametros_2ck_default, G_2CK, G_2CK_T, entropia_residual,
)
from src.rede_pinn_2ck import RedePINN_G, RedePINN_Rho
from src.treinamento_2ck import treinar_G, treinar_rho


def principal():
    print("=" * 70)
    print("  2CK · Não-Fermi Líquido · S_res = ½ k_B ln 2 · PINN")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_2ck_default()
    print(f"\n[1] Device: {dev}, T_K={p['T_K']}, S_res={entropia_residual(p):.4f}")

    # dados G(V,T)
    n_pts = 400
    V = np.random.uniform(-3, 3, n_pts)
    T = np.random.uniform(0.05, 2.0, n_pts)
    G_ref = G_2CK(V, T, p)
    VT = torch.tensor(np.column_stack([V, T]), dtype=torch.float32, device=dev)
    G_t = torch.tensor(G_ref, dtype=torch.float32, device=dev).reshape(-1, 1)

    print("\n[2] Treinando PINN G(V,T)...")
    rede_G = RedePINN_G().to(dev)
    res_G = treinar_G(rede_G, VT, G_t, n_epocas=2000, taxa=1e-3, verbose_cada=200)
    print(f"    Perda G final: {res_G['perda_final']:.4e}")

    # ρ(t)
    t = torch.linspace(0, 8, 120, device=dev).reshape(-1, 1).requires_grad_(True)
    print("\n[3] Treinando PINN ρ(t)...")
    rede_rho = RedePINN_Rho().to(dev)
    res_rho = treinar_rho(rede_rho, t, p, n_epocas=2000, taxa=1e-3, verbose_cada=200)
    print(f"    Perda ρ final: {res_rho['perda_final']:.4e}")

    # curvas
    T_line = np.linspace(0.02, 2.5, 100)
    G_T = G_2CK_T(T_line, p)
    VT0 = torch.tensor(np.column_stack([np.zeros_like(T_line), T_line]), dtype=torch.float32, device=dev)
    with torch.no_grad():
        G_nn = rede_G(VT0).cpu().numpy().ravel()

    V_line = np.linspace(-3, 3, 100)
    G_V = G_2CK(V_line, np.full_like(V_line, 0.2), p)

    with torch.no_grad():
        a, c, d, b = rede_rho.matriz_rho(t)
    t_np = t.detach().cpu().numpy().ravel()
    a_np = a.cpu().numpy().ravel()

    print("\n[4] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(T_line, G_T, "k--", lw=2, label="2CK analítico")
    ax.plot(T_line, G_nn, "C0-", lw=1.5, label="PINN")
    ax.set_xlabel(r"$T/T_K$"); ax.set_ylabel(r"$G$")
    ax.set_title(r"(a) $G(T)=G_{max}[1-A\sqrt{T/T_K}]$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(V_line, G_V, "C3-", lw=2)
    ax.set_xlabel(r"$eV / k_B T_K$"); ax.set_ylabel(r"$G$")
    ax.set_title(r"(b) $G(V)$ a $T=0.2\,T_K$ (NFL)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(t_np, a_np, "C2-", lw=2, label=r"$\rho_{\uparrow\uparrow}$")
    ax.axhline(0.5, color="k", ls="--", label="eq. 1/2")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\rho$")
    ax.set_title(r"(c) Dinâmica $\hat\rho_d(t)$ (spin)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res_G["historico"], "C0-", lw=1, label="G")
    ax.semilogy(res_rho["historico"], "C2-", lw=1, label=r"$\rho$")
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "2ck_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print(f"    S_res / k_B = {entropia_residual(p):.4f}  (½ ln 2)")
    print("\n" + "=" * 70)
    print("  2CK NFL concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
