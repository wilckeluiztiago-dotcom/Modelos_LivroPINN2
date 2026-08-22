#!/usr/bin/env python3
"""
Cross-Tunneling Leakage · Nanofios Acoplados · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_tunel import parametros_tunel_default, G_tunel_wkb, J_leak
from src.rede_pinn_tunel import RedePINN_Tunel
from src.treinamento_tunel import treinar_tunel


def principal():
    print("=" * 70)
    print("  Cross-Tunneling Leakage · Inter-Fio · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_tunel_default()
    G_eff = p["G_leak0"] * G_tunel_wkb(p["d_int"], p["Phi_B"])
    print(f"\n[1] Device: {dev}")
    print(f"    d_int={p['d_int']}, Φ_B={p['Phi_B']}, G_eff={G_eff:.4e}")

    # colocation (z,t) ∈ [0,1]×[0,1]
    n_col = 400
    z = torch.rand(n_col, 1, device=dev)
    t = torch.rand(n_col, 1, device=dev)
    zt_col = torch.cat([z, t], dim=1).requires_grad_(True)

    # BC em z=0: polarização diferente nos dois fios
    n_bc = 50
    t_bc = torch.linspace(0, 1, n_bc, device=dev).reshape(-1, 1)
    z0 = torch.zeros(n_bc, 1, device=dev)
    zt_bc = torch.cat([z0, t_bc], dim=1)
    # V1(0,t)=1, V2(0,t)=0  → força diferença que gera J_leak
    V_bc = torch.cat([torch.ones(n_bc, 1, device=dev), torch.zeros(n_bc, 1, device=dev)], dim=1)

    print("\n[2] Treinando PINN (V1,V2,I1,I2)...")
    rede = RedePINN_Tunel([2, 48, 48, 48, 4]).to(dev)
    npar = sum(q.numel() for q in rede.parameters())
    print(f"    Parâmetros: {npar}")
    res = treinar_tunel(
        rede, zt_col, zt_bc, V_bc, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfis em t fixo
    zg = torch.linspace(0, 1, 100, device=dev).reshape(-1, 1)
    tg = torch.full_like(zg, 0.5)
    zt = torch.cat([zg, tg], dim=1)
    with torch.no_grad():
        V1, V2, I1, I2 = rede.campos(zt)
    z_np = zg.cpu().numpy().ravel()
    V1n = V1.cpu().numpy().ravel()
    V2n = V2.cpu().numpy().ravel()
    I1n = I1.cpu().numpy().ravel()
    I2n = I2.cpu().numpy().ravel()
    Jl = G_eff * (V1n - V2n)

    # varredura d_int
    ds = np.linspace(0.4, 2.0, 40)
    G_vs_d = [parametros_tunel_default()["G_leak0"] * G_tunel_wkb(d, p["Phi_B"]) for d in ds]

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(z_np, V1n, "C0-", lw=2, label=r"$V_1$")
    ax.plot(z_np, V2n, "C3-", lw=2, label=r"$V_2$")
    ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$V$")
    ax.set_title(r"(a) Potenciais nos nanofios ($t=0.5$)")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(z_np, Jl, "C2-", lw=2)
    ax.set_xlabel(r"$z$"); ax.set_ylabel(r"$J_{leak}$")
    ax.set_title(r"(b) Densidade de corrente de tunelamento")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.semilogy(ds, G_vs_d, "k-", lw=2)
    ax.axvline(p["d_int"], color="C3", ls="--", label=fr"$d_{{int}}={p['d_int']}$")
    ax.set_xlabel(r"$d_{\mathrm{int}}$"); ax.set_ylabel(r"$G_{\mathrm{eff}}$")
    ax.set_title(r"(c) Exponencial WKB vs distância inter-fio")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "cross_tunnel_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  Cross-tunneling leakage concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
