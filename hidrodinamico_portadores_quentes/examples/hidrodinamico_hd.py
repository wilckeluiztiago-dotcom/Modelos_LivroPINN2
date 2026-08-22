#!/usr/bin/env python3
"""
Modelo Hidrodinâmico de Portadores Quentes · Baccarani–Wordeman · PINN
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_hd import parametros_hd_default, perfil_campo_1d, W_n, W0
from src.rede_pinn_hd import RedePINN_HD
from src.treinamento_hd import treinar_hd


def principal():
    print("=" * 70)
    print("  Hidrodinâmico · Portadores Quentes · Velocity Overshoot · PINN")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_hd_default()
    print(f"\n[1] Device: {dev}, E_field={p['E_field']}, T0={p['T0']}")

    n_col = 200
    x_col = torch.linspace(0.02, 0.98, n_col, device=dev).reshape(-1, 1).requires_grad_(True)

    # BC: n e Tn nas bordas
    x_bc = torch.tensor([[0.0], [1.0]], device=dev)
    n_bc = torch.tensor([[1.0], [1.0]], device=dev)
    Tn_bc = torch.tensor([[p["T0"]], [p["T0"]]], device=dev)

    print("\n[2] Treinando PINN (n, v, T_n)...")
    rede = RedePINN_HD([1, 48, 48, 48, 3]).to(dev)
    res = treinar_hd(
        rede, x_col, x_bc, n_bc, Tn_bc, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    xg = torch.linspace(0, 1, 150, device=dev).reshape(-1, 1)
    with torch.no_grad():
        n, v, Tn = rede.campos(xg)
    x_np = xg.cpu().numpy().ravel()
    n_np = n.cpu().numpy().ravel()
    v_np = v.cpu().numpy().ravel()
    Tn_np = Tn.cpu().numpy().ravel()
    E_np = perfil_campo_1d(x_np, p)
    # velocidade de saturação drift-diffusion aproximada vs overshoot
    v_dd = p["E_field"] * p["tau_p0"] / p["m_star"] * np.ones_like(x_np) * 0.3

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x_np, n_np, "C0-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$n(x)$")
    ax.set_title("(a) Densidade de portadores")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(x_np, v_np, "C3-", lw=2, label=r"$v_n$ HD")
    ax.plot(x_np, v_dd, "k--", lw=1.5, label=r"$v$ DD ref.")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$v$")
    ax.set_title(r"(b) Velocity overshoot ($v_n$ vs DD)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(x_np, Tn_np, "C2-", lw=2, label=r"$T_n$")
    ax.axhline(p["T0"], color="k", ls="--", label=r"$T_L$ (rede)")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$T$")
    ax.set_title(r"(c) Temperatura eletrônica vs rede")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "hd_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print(f"    T_n max / T0 = {Tn_np.max()/p['T0']:.3f}")
    print("\n" + "=" * 70)
    print("  Hidrodinâmico portadores quentes concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
