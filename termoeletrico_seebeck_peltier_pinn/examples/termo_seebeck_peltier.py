#!/usr/bin/env python3
"""
Transporte Termoelétrico Não-Linear · Seebeck / Peltier · PINN (PyTorch)
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_termo import parametros_termo_default
from src.rede_pinn_termo import RedePINN_Termo
from src.residuo_termo import residuos_termo
from src.treinamento_termo import treinar_termo


def principal():
    print("=" * 70)
    print("  Termoelétrico · Seebeck / Peltier · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    print(f"\n[1] Device: {dev}")
    p = parametros_termo_default()
    print(f"    σ={p['sigma']}, S={p['S']}, κ={p['kappa']}")

    # domínio x ∈ [0, 1]  (fonte → dreno)
    n_col = 200
    x_col = torch.linspace(0, 1, n_col, device=dev).reshape(-1, 1).requires_grad_(True)

    # BC: φ(0)=0, φ(1)=V_bias; T(0)=T_s, T(1)=T_d
    V_bias, T_s, T_d = 0.5, 1.2, 0.9
    x_bc = torch.tensor([[0.0], [1.0]], device=dev)
    phi_bc = torch.tensor([[0.0], [V_bias]], device=dev)
    T_bc = torch.tensor([[T_s], [T_d]], device=dev)

    print("\n[2] Treinando PINN (φ, T)...")
    rede = RedePINN_Termo(camadas=[1, 48, 48, 48, 2]).to(dev)
    n_params = sum(p_.numel() for p_ in rede.parameters())
    print(f"    Parâmetros: {n_params}")
    res = treinar_termo(
        rede, x_col, x_bc, phi_bc, T_bc, p,
        n_epocas=2500, taxa=1e-3, verbose_cada=250,
    )
    print(f"    Perda final: {res['perda_final']:.4e}")

    # perfil
    x_plot = torch.linspace(0, 1, 150, device=dev).reshape(-1, 1).requires_grad_(True)
    with torch.enable_grad():
        phi, T = rede.campos(x_plot)
        R_c, R_e = residuos_termo(rede, x_plot, p)
        dphi = torch.autograd.grad(phi, x_plot, torch.ones_like(phi), create_graph=True)[0]
        dT = torch.autograd.grad(T, x_plot, torch.ones_like(T), create_graph=True)[0]
        J = -p["sigma"] * dphi - p["sigma"] * p["S"] * dT

    x_np = x_plot.detach().cpu().numpy().ravel()
    phi_np = phi.detach().cpu().numpy().ravel()
    T_np = T.detach().cpu().numpy().ravel()
    J_np = J.detach().cpu().numpy().ravel()

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(x_np, phi_np, "C0-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$\phi(x)$")
    ax.set_title(r"(a) Potencial eletrostático $\phi$")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(x_np, T_np, "C3-", lw=2)
    ax.axhline(T_s, color="C3", ls=":", alpha=0.5)
    ax.axhline(T_d, color="C0", ls=":", alpha=0.5)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$T(x)$")
    ax.set_title(r"(b) Temperatura (Peltier / Seebeck)")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    ax.plot(x_np, J_np, "C2-", lw=2)
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$J(x)$")
    ax.set_title(r"(c) Corrente $J=-\sigma\nabla\phi-\sigma S\nabla T$")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN PyTorch")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "termo_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")

    print("\n" + "=" * 70)
    print("  Termoelétrico Seebeck/Peltier concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
