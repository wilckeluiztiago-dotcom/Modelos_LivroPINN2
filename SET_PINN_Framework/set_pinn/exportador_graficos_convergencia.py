# -*- coding: utf-8 -*-
"""
Módulo 31: Exportação de Gráficos (Diamantes, Densidades, Potenciais)
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from pathlib import Path

def salvar_mapa_diamantes(
    I: torch.Tensor,
    V_D: torch.Tensor,
    V_G: torch.Tensor,
    caminho: str = "diamantes_coulomb.png"
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 7))
        plt.pcolormesh(
            V_G.detach().cpu().numpy(),
            V_D.detach().cpu().numpy(),
            I.detach().cpu().numpy(),
            shading="auto",
            cmap="inferno"
        )
        plt.colorbar(label="Corrente I [A]")
        plt.xlabel(r"$V_G$ [V]")
        plt.ylabel(r"$V_D$ [V]")
        plt.title("Diamantes de Coulomb – PINN SET\nAutor: Luiz Tiago Wilcke")
        plt.tight_layout()
        plt.savefig(caminho, dpi=200)
        plt.close()
        print(f"[OK] Gráfico salvo em: {caminho}")
    except Exception as e:
        print(f"[Aviso] Não foi possível gerar gráfico: {e}")
