"""
Script Principal – Nanotransistor 2 nm Si:P via PINNs (versão física reforçada)
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

from producao_solver_modular import SolverPINNNanotransistor
from amostragem_lhs import amostragem_lhs


def main():
    print("=" * 72)
    print("  PINN Nanotransistor 2 nm Si:P  –  Física reforçada")
    print("  Autor: Luiz Tiago Wilcke")
    print("=" * 72)

    device = "cpu"
    results_dir = Path(__file__).parent.parent / "results"
    figures_dir = Path(__file__).parent.parent / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    solver = SolverPINNNanotransistor(Vds=0.5, T=300.0, device=device)
    print(solver.geo.resumo())
    print(solver.mat.resumo())
    print(f"λ² (Debye) ≈ {solver.perda_fn.res_poisson.lambda2:.3e}")

    print("\n>>> Treinamento híbrido (Adam + L-BFGS)...")
    historico = solver.treinar(epochs_adam=900, epochs_lbfgs=40, n_col=1800)

    # --- Predições físicas ---
    x_plot = torch.linspace(0, 1, 400, device=device).unsqueeze(1)
    phi_V = solver.potencial_V(x_plot).cpu().numpy().flatten()
    n_m3 = solver.densidade_m3(x_plot).cpu().numpy().flatten()
    Nd = solver.perfil(x_plot.cpu()).numpy().flatten()
    J = solver.corrente_aproximada(x_plot).detach().cpu().numpy().flatten()

    # --- Gráficos ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    axes[0, 0].plot(x_plot.cpu().numpy(), phi_V, "b-", lw=2.2)
    axes[0, 0].set_xlabel("x / L  (Fonte → Dreno)")
    axes[0, 0].set_ylabel("φ (V)")
    axes[0, 0].set_title("Potencial eletrostático")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(x_plot.cpu().numpy(), Nd * 1e-6, "r-", lw=2.2, label="N_D")
    axes[0, 1].semilogy(x_plot.cpu().numpy(), n_m3 * 1e-6 + 1e12, "g--", lw=1.8, label="n")
    axes[0, 1].set_xlabel("x / L")
    axes[0, 1].set_ylabel("Concentração (cm$^{-3}$)")
    axes[0, 1].set_title("Dopagem de fósforo e densidade de elétrons")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(x_plot.cpu().numpy(), J * 1e-6, "m-", lw=2)
    axes[1, 0].set_xlabel("x / L")
    axes[1, 0].set_ylabel("J_n (µA/µm) [proxy]")
    axes[1, 0].set_title("Densidade de corrente de elétrons")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].semilogy(historico, "k-", alpha=0.85)
    axes[1, 1].set_xlabel("Época / Step")
    axes[1, 1].set_ylabel("Loss total")
    axes[1, 1].set_title("Curva de aprendizado (Adam → L-BFGS)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle("Nanotransistor 2 nm Si:P – PINN (Luiz Tiago Wilcke)", fontsize=13)
    plt.tight_layout()
    fig.savefig(figures_dir / "resultados_completos_pinn.png", dpi=160, bbox_inches="tight")
    print(f"\nGráfico salvo: {figures_dir / 'resultados_completos_pinn.png'}")

    solver.salvar(str(results_dir / "modelo_nanotransistor_completo.pt"))
    np.savez(results_dir / "dados_numericos_completos.npz",
             x=x_plot.cpu().numpy().flatten(),
             phi_V=phi_V, n_m3=n_m3, Nd=Nd, J=J,
             loss_history=np.array(historico))

    print(f"Modelo e dados salvos em {results_dir}")
    print("Concluído.")
    return solver, historico


if __name__ == "__main__":
    main()
