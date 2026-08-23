"""
Script Principal de Treinamento - Nanotransistor 2 nm Si:P via PINNs
Autor: Luiz Tiago Wilcke
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from geometria_dispositivo import GeometriaNanotransistor
from parametros_materiais_si import ParametrosSilicio
from perfil_dopagem_fosforo import PerfilDopagemFosforo
from arquitetura_pinn_poderosa import PINNPoderosa
from funcao_perda_composta import PerdaCompostaPINN
from otimizacao_hibrida import treinar_adam, treinar_lbfgs
from amostragem_lhs import amostragem_lhs
from producao_solver_modular import SolverPINNNanotransistor


def main():
    print("=" * 70)
    print("  PINN Nanotransistor 2 nm Si:P  –  Luiz Tiago Wilcke")
    print("=" * 70)

    device = "cpu"
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # --- Solver modular completo ---
    solver = SolverPINNNanotransistor(device=device)
    print(solver.geo.resumo())
    print(solver.mat.resumo())
    print(f"Parâmetros da rede: {solver.modelo.num_parametros():,}")

    # --- Treinamento ---
    print("\n>>> Iniciando treinamento híbrido (Adam + L-BFGS)...")
    historico = solver.treinar(epochs_adam=800, epochs_lbfgs=40, n_col=1500)

    # --- Predição e gráficos ---
    x_plot = torch.linspace(0, 1, 300, device=device).unsqueeze(1)
    with torch.no_grad():
        saida = solver.prever(x_plot)
        phi = saida[:, 0].cpu().numpy()
        n = torch.abs(saida[:, 1]).cpu().numpy()
        Nd = solver.perfil(x_plot.cpu()).numpy().flatten()

    # Gráficos
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(x_plot.cpu().numpy(), phi, "b-", lw=2)
    axes[0, 0].set_xlabel("x normalizado (Fonte → Dreno)")
    axes[0, 0].set_ylabel("Potencial φ (V)")
    axes[0, 0].set_title("Potencial Eletrostático (PINN)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(x_plot.cpu().numpy(), Nd * 1e-6, "r-", lw=2)
    axes[0, 1].set_xlabel("x normalizado")
    axes[0, 1].set_ylabel("N_D (cm$^{-3}$)")
    axes[0, 1].set_title("Perfil de Dopagem de Fósforo")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].semilogy(x_plot.cpu().numpy(), n * 1e-6 + 1e10, "g-", lw=2)
    axes[1, 0].set_xlabel("x normalizado")
    axes[1, 0].set_ylabel("n (cm$^{-3}$) [proxy]")
    axes[1, 0].set_title("Densidade de Elétrons (saída da rede)")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(historico, "k-", alpha=0.8)
    axes[1, 1].set_xlabel("Época / Step")
    axes[1, 1].set_ylabel("Loss total")
    axes[1, 1].set_title("Curva de Aprendizado (Adam + L-BFGS)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(figures_dir / "resultados_completos_pinn.png", dpi=150, bbox_inches="tight")
    print(f"\nGráfico salvo: {figures_dir / 'resultados_completos_pinn.png'}")

    # Salvar modelo e dados
    solver.salvar(results_dir / "modelo_nanotransistor_completo.pt")
    np.savez(results_dir / "dados_numericos_completos.npz",
             x=x_plot.cpu().numpy().flatten(),
             phi=phi, n=n, Nd=Nd,
             loss_history=np.array(historico))

    print(f"Modelo e dados salvos em {results_dir}")
    print("Treinamento concluído com sucesso.")
    return solver, historico


if __name__ == "__main__":
    main()
