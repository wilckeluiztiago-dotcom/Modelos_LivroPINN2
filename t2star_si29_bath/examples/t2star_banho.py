#!/usr/bin/env python3
"""
T₂* Dephasing · ³¹P + banho ²⁹Si · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_banho import (
    parametros_banho_default, gerar_banho_29Si,
    acoplamentos_dipolares, T2_star_de_A, fid_gaussiano,
)
from src.rede_pinn_t2 import RedePINN_T2
from src.treinamento_t2 import treinar_t2


def principal():
    print("=" * 70)
    print("  T₂* · ³¹P + banho ²⁹Si (4.7%) · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_banho_default()
    print(f"\n[1] Device: {dev}, c_29Si={p['c_29Si']}, N_bath={p['N_bath']}")

    pos = gerar_banho_29Si(p["N_bath"], p["R_max"], semente=42)
    A = acoplamentos_dipolares(pos, p["gamma_P"], p["gamma_29"], p["mu0_hbar"])
    T2s = T2_star_de_A(A)
    print(f"    T₂* físico (Σ A_k²) = {T2s:.4f}")

    n_col = 200
    t_col = torch.linspace(0.01, 3.0 * T2s, n_col, device=dev).reshape(-1, 1).requires_grad_(True)

    print("\n[2] Treinando PINN ⟨S_x(t)⟩...")
    rede = RedePINN_T2([1, 48, 48, 1]).to(dev)
    # inicializa log_T2s perto do valor físico
    with torch.no_grad():
        rede.log_T2s.copy_(torch.log(torch.tensor(T2s, device=dev)))
    res = treinar_t2(rede, t_col, T2s, n_epocas=2000, taxa=1e-3, verbose_cada=200)
    print(f"    Perda final: {res['perda_final']:.4e}")
    print(f"    T₂*_θ = {float(rede.T2_star().detach()):.4f}")

    t_np = np.linspace(0, 3.0 * T2s, 200)
    fid_ref = fid_gaussiano(t_np, T2s)
    t_t = torch.tensor(t_np, dtype=torch.float32, device=dev).reshape(-1, 1)
    with torch.no_grad():
        fid_nn = rede(t_t).cpu().numpy().ravel()

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(t_np, fid_ref, "k--", lw=2, label=r"$\exp(-(t/T_2^*)^2)$")
    ax.plot(t_np, fid_nn, "C0-", lw=2, label="PINN")
    ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$\langle S_x\rangle$")
    ax.set_title(r"(a) FID gaussiano / dephasing $T_2^*$")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.hist(A, bins=30, color="C3", alpha=0.8)
    ax.set_xlabel(r"$A_k^{\mathrm{dip}}$"); ax.set_ylabel("contagem")
    ax.set_title(r"(b) Distribuição de acoplamentos dipolares")
    ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    r = np.linalg.norm(pos, axis=1)
    sc = ax.scatter(pos[:, 0], pos[:, 1], c=r, cmap="viridis", s=20)
    plt.colorbar(sc, ax=ax, fraction=0.046, label=r"$r$")
    ax.plot(0, 0, "r*", ms=12, label=r"$^{31}$P")
    ax.set_xlabel(r"$x$"); ax.set_ylabel(r"$y$")
    ax.set_title(r"(c) Banho $^{29}$Si (projeção xy)")
    ax.legend(fontsize=8); ax.set_aspect("equal")

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "t2star_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  T₂* ²⁹Si bath concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
