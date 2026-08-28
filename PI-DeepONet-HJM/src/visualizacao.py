"""
Módulo 15: Visualização de Curvas e Superfícies
Autor: Luiz Tiago Wilcke
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from .config import CONFIG
from .avaliacao import avaliar_superficie
from .arquitetura_deeponet import PIDeepONetHJM
from .geracao_curvas import gerar_ensemble_curvas


def plotar_superficie_precos(
    modelo: PIDeepONetHJM,
    u: np.ndarray = None,
    caminho: str = "results/superficie_P.png",
):
    """Gera e salva a superfície P(t,T)."""
    if u is None:
        u_t, _ = gerar_ensemble_curvas(num_curvas=1)
        u = u_t[0:1]

    dados = avaliar_superficie(modelo, u)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(dados["t"], dados["T"], dados["P"], cmap="viridis", alpha=0.9)
    ax.set_xlabel("t (anos)")
    ax.set_ylabel("T (anos)")
    ax.set_zlabel("P(t,T)")
    ax.set_title("Superfície de Preços de Títulos Zero-Cupom – PI-DeepONet HJM")
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Superfície salva em {caminho}")


def plotar_curva_forward_inicial(u: np.ndarray, T_sensores: np.ndarray, caminho: str = "results/curva_forward.png"):
    """Plota a curva forward inicial."""
    plt.figure(figsize=(8, 4))
    plt.plot(T_sensores, u.squeeze(), "b-o", markersize=3)
    plt.xlabel("Maturidade T (anos)")
    plt.ylabel("f(0, T)")
    plt.title("Curva Forward Inicial")
    plt.grid(True, alpha=0.3)
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()


def plotar_historico_perda(historico: dict, caminho: str = "results/historico_perda.png"):
    """Plota a evolução da perda."""
    plt.figure(figsize=(9, 5))
    plt.semilogy(historico["total"], label="Total")
    if "fisica" in historico:
        plt.semilogy(historico["fisica"], label="Física (EDP)")
    if "dados" in historico:
        plt.semilogy(historico["dados"], label="Dados (CI)")
    plt.xlabel("Época")
    plt.ylabel("Perda (log)")
    plt.legend()
    plt.title("Evolução da Perda – PI-DeepONet HJM")
    plt.grid(True, alpha=0.3)
    Path(caminho).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close()
