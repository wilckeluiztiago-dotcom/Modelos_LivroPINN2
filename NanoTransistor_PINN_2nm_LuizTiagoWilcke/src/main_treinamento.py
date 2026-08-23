"""
Main Training Script - Nanotransistor 2 nm PINN
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from geometria_dispositivo import GeometriaNanotransistor
from parametros_materiais_si import ParametrosSilicio
from perfil_dopagem_fosforo import PerfilDopagemFosforo
from equacao_poisson import ResidualPoisson
from equacao_schrodinger import ResidualSchrodinger
from arquitetura_pinn_poderosa import PINNPoderosa

def amostragem_lhs(n_pontos, dim=1):
    """Latin Hypercube Sampling simplificado."""
    try:
        from scipy.stats import qmc
        sampler = qmc.LatinHypercube(d=dim)
        return torch.tensor(sampler.random(n=n_pontos), dtype=torch.float32)
    except Exception:
        return torch.rand(n_pontos, dim)

def treinar_pinn_poisson_1d(epochs_adam=1500, epochs_lbfgs=100, device="cpu"):
    print("="*60)
    print("Treinamento PINN - Poisson 1D para Nanotransistor 2 nm")
    print("Autor: Luiz Tiago Wilcke")
    print("="*60)

    geo = GeometriaNanotransistor()
    mat = ParametrosSilicio()
    perfil = PerfilDopagemFosforo(geo, mat)
    residual_p = ResidualPoisson(mat)

    modelo = PINNPoderosa(in_dim=1, out_dim=1, hidden=128, n_blocks=4).to(device)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(otimizador, step_size=400, gamma=0.5)

    n_col = 1500
    x_col = amostragem_lhs(n_col, 1).to(device).requires_grad_(True)

    Vds = 0.5
    x_bc_s = torch.zeros(150, 1, device=device, requires_grad=True)
    x_bc_d = torch.ones(150, 1, device=device, requires_grad=True)

    historico_loss = []

    for ep in range(epochs_adam):
        otimizador.zero_grad()
        phi = modelo(x_col)
        Nd = perfil(x_col.detach())
        VT = mat.VT()
        # Boltzmann approx para equilíbrio
        n = Nd * torch.exp(torch.clamp(phi / VT, -20, 20))
        p = torch.zeros_like(n)
        Na = torch.zeros_like(Nd)

        res = residual_p.residual(phi, n, p, Nd, Na, x_col)
        loss_pde = residual_p.perda(res)

        phi_s = modelo(x_bc_s)
        phi_d = modelo(x_bc_d)
        loss_bc = torch.mean(phi_s**2) + torch.mean((phi_d - Vds)**2)

        loss = loss_pde + 20.0 * loss_bc
        loss.backward()
        torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        otimizador.step()
        scheduler.step()

        historico_loss.append(loss.item())
        if ep % 150 == 0:
            print(f"Epoch {ep:5d} | Loss total: {loss.item():.3e} | PDE: {loss_pde.item():.3e} | BC: {loss_bc.item():.3e}")

    print("\n--- Refinamento L-BFGS ---")
    otimizador_lbfgs = torch.optim.LBFGS(modelo.parameters(), lr=0.3, max_iter=15,
                                         history_size=30, line_search_fn="strong_wolfe")

    def closure():
        otimizador_lbfgs.zero_grad()
        phi = modelo(x_col)
        Nd = perfil(x_col.detach())
        n = Nd * torch.exp(torch.clamp(phi / mat.VT(), -20, 20))
        p = torch.zeros_like(n)
        Na = torch.zeros_like(Nd)
        res = residual_p.residual(phi, n, p, Nd, Na, x_col)
        loss_pde = residual_p.perda(res)
        phi_s = modelo(x_bc_s)
        phi_d = modelo(x_bc_d)
        loss_bc = torch.mean(phi_s**2) + torch.mean((phi_d - Vds)**2)
        loss = loss_pde + 20.0 * loss_bc
        loss.backward()
        return loss

    for i in range(max(1, epochs_lbfgs // 15)):
        loss_val = otimizador_lbfgs.step(closure)
        print(f"LBFGS step {i} | Loss: {float(loss_val):.3e}")

    # Resultados
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    figures_dir = Path(__file__).parent.parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    x_plot = torch.linspace(0, 1, 300, device=device).unsqueeze(1)
    with torch.no_grad():
        phi_plot = modelo(x_plot).cpu().numpy().flatten()
        Nd_plot = perfil(x_plot.cpu()).numpy().flatten()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].plot(x_plot.cpu().numpy(), phi_plot, "b-", lw=2)
    axes[0].set_xlabel("x normalizado (Fonte → Dreno)")
    axes[0].set_ylabel("Potencial φ (V)")
    axes[0].set_title("Potencial Eletrostático PINN\nNanotransistor 2 nm Si:P")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(x_plot.cpu().numpy(), Nd_plot * 1e-6, "r-", lw=2)
    axes[1].set_xlabel("x normalizado")
    axes[1].set_ylabel("N_D (cm⁻³)")
    axes[1].set_title("Perfil de Dopagem de Fósforo")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(historico_loss, "g-", alpha=0.8)
    axes[2].set_xlabel("Época")
    axes[2].set_ylabel("Loss total")
    axes[2].set_title("Curva de Aprendizado (Adam + L-BFGS)")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(figures_dir / "resultados_poisson_1d.png", dpi=150, bbox_inches="tight")
    print(f"\nGráfico salvo em {figures_dir / 'resultados_poisson_1d.png'}")

    torch.save(modelo.state_dict(), results_dir / "modelo_poisson_1d.pt")
    np.savez(results_dir / "dados_numericos.npz",
             x=x_plot.cpu().numpy().flatten(),
             phi=phi_plot,
             Nd=Nd_plot,
             loss_history=np.array(historico_loss))

    print("Treinamento concluído com sucesso.")
    print(f"Loss final: {historico_loss[-1]:.3e}")
    return modelo, historico_loss

if __name__ == "__main__":
    treinar_pinn_poisson_1d()
