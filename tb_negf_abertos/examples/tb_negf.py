#!/usr/bin/env python3
"""
TB-NEGF Atomístico · Eletrodos Abertos · PINN PyTorch
Autor: Luiz Tiago Wilcke
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.utils import set_seed, device_padrao
from src.fisica_negf import (
    parametros_negf_default, hamiltoniano_canal, green_retardada,
    transmissao, corrente_landauer,
)
from src.rede_pinn_negf import RedePINN_NEGF
from src.treinamento_negf import treinar_negf


def principal():
    print("=" * 70)
    print("  TB-NEGF · Si:P + leads n+ · PINN PyTorch")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 70)

    set_seed(42)
    dev = device_padrao()
    p = parametros_negf_default()
    H = hamiltoniano_canal(p)
    n = H.shape[0]
    print(f"\n[1] Device: {dev}, n_sites={n}, V_P={p['V_P']}")

    I, E_grid, T_grid = corrente_landauer(H, p)
    print(f"    I_ds (Landauer) = {I:.4f} (u.a. 2e/h=1)")
    print(f"    T max = {T_grid.max():.4f}")

    # batch de energias para treino
    n_E = 40
    E_batch = torch.linspace(-3, 3, n_E, device=dev).reshape(-1, 1)

    print("\n[2] Treinando PINN G^R(E)...")
    rede = RedePINN_NEGF(n=n, camadas=[1, 64, 64, 2 * n * n]).to(dev)
    res = treinar_negf(rede, E_batch, H, p, n_epocas=2500, taxa=1e-3, verbose_cada=250)
    print(f"    Perda final: {res['perda_final']:.4e}")

    # comparar G^R diag Im
    E_test = np.linspace(-3, 3, 80)
    im_diag_ex = []
    im_diag_nn = []
    for e in E_test:
        GR = green_retardada(e, H, p)
        im_diag_ex.append(np.mean(np.diag(GR.imag)))
        with torch.no_grad():
            Et = torch.tensor([[e]], dtype=torch.float32, device=dev)
            _, Im = rede(Et)
            im_diag_nn.append(float(torch.mean(torch.diag(Im[0])).cpu()))
    im_diag_ex = np.array(im_diag_ex)
    im_diag_nn = np.array(im_diag_nn)

    print("\n[3] Figuras...")
    fig, eixos = plt.subplots(2, 2, figsize=(12, 9))

    ax = eixos[0, 0]
    ax.plot(E_grid, T_grid, "C0-", lw=2)
    ax.set_xlabel(r"$E$"); ax.set_ylabel(r"$\mathcal{T}(E)$")
    ax.set_title(r"(a) Transmissão Landauer $\mathcal{T}(E)$")
    ax.grid(True, alpha=0.3)

    ax = eixos[0, 1]
    ax.plot(E_test, im_diag_ex, "k--", lw=2, label="NEGF exact")
    ax.plot(E_test, im_diag_nn, "C3-", lw=1.5, label="PINN")
    ax.set_xlabel(r"$E$"); ax.set_ylabel(r"$\langle\mathrm{Im}\,G^R\rangle$")
    ax.set_title(r"(b) $\mathrm{Im}\,G^R$ (densidade de estados)")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 0]
    # LDOS no centro vs energia
    centro = n // 2
    ldos = []
    for e in E_test:
        GR = green_retardada(e, H, p)
        ldos.append(-GR[centro, centro].imag / np.pi)
    ax.plot(E_test, ldos, "C2-", lw=2)
    ax.axvline(p["V_P"], color="gray", ls=":", label=r"$V_P$")
    ax.set_xlabel(r"$E$"); ax.set_ylabel("LDOS centro")
    ax.set_title(r"(c) LDOS no sítio $^{31}$P")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = eixos[1, 1]
    ax.semilogy(res["historico"], "C4-", lw=1.2)
    ax.set_xlabel("época"); ax.set_ylabel("perda")
    ax.set_title("(d) Treinamento PINN Dyson")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    caminho = os.path.join(os.path.dirname(__file__), "..", "figures", "negf_result.png")
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    print(f"    Figura: {caminho}")
    print("\n" + "=" * 70)
    print("  TB-NEGF concluído.")
    print("=" * 70)


if __name__ == "__main__":
    principal()
