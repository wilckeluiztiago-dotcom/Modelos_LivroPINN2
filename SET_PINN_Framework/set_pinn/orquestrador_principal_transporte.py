# -*- coding: utf-8 -*-
"""
Módulo 32: Orquestrador Principal do Framework PINN-SET
Autor: Luiz Tiago Wilcke
"""

from __future__ import annotations
import torch
from .configuracao_dispositivo import criar_configuracao_padrao
from .amostragem_hipercubo_latino import amostragem_lhs
from .pinn_fokker_planck_continua import PINNFokkerPlanck
from .otimizador_hibrido_treinamento import treinar_hibrido
from .gerador_diamantes_coulomb import mapa_corrente
from .exportador_graficos_convergencia import salvar_mapa_diamantes
from .monitoramento_tensorboard import MonitorTensorBoard
from .constantes_fisicas import DTYPE, DEVICE

def main() -> None:
    print("=" * 72)
    print("  Framework PINN para Transporte Quântico em SET")
    print("  Autor: Luiz Tiago Wilcke")
    print("  Baseado no Volume II – Redes Neurais Informadas pela Física")
    print("=" * 72)
    print(f"Dispositivo de computação: {DEVICE}")
    print()

    cfg = criar_configuracao_padrao()
    print(f"Energia de carregamento E_C = {cfg.E_C.item():.4e} J")
    print(f"C_Σ = {cfg.C_Sigma.item():.4e} F")
    print()

    monitor = MonitorTensorBoard(log_dir="runs/set_pinn")

    # Amostragem LHS
    limites = {
        "t": (0.0, 1e-6),
        "q": (-5 * cfg.E_C.item(), 5 * cfg.E_C.item()),
        "V_D": (-0.05, 0.05),
        "V_G": (-0.2, 0.2)
    }
    print("Gerando pontos de colocalização via Latin Hypercube Sampling...")
    pts = amostragem_lhs(2048, limites)
    print(f"  → {len(pts['t'])} pontos gerados.")

    # Modelo principal
    modelo = PINNFokkerPlanck(camadas=[64, 64, 64]).to(DEVICE)
    print(f"Modelo PINN-Fokker-Planck criado com {sum(p.numel() for p in modelo.parameters())} parâmetros.")
    print()

    def perda_total() -> torch.Tensor:
        entradas = torch.stack(
            [pts["t"], pts["q"], pts["V_D"], pts["V_G"]], dim=-1
        ).requires_grad_(True)
        D1 = 0.05 * pts["V_D"].unsqueeze(-1)
        D2 = 0.01 * torch.ones_like(pts["V_D"]).unsqueeze(-1)
        res = modelo.residuo_fp(entradas, D1, D2)
        return torch.mean(res ** 2)

    print("Iniciando treinamento híbrido (Adam + L-BFGS)...")
    treinar_hibrido(modelo, perda_total, n_adam=300, n_lbfgs=20, lr_adam=1e-3)
    print()

    # Mapa de diamantes de Coulomb
    print("Gerando mapa de Diamantes de Coulomb...")
    VD = torch.linspace(-0.04, 0.04, 80, dtype=DTYPE, device=DEVICE)
    VG = torch.linspace(-0.15, 0.15, 80, dtype=DTYPE, device=DEVICE)
    I = mapa_corrente(cfg, VD, VG, n_max=2)
    salvar_mapa_diamantes(I, VD, VG, caminho="diamantes_coulomb.png")

    monitor.fechar()
    print()
    print("=" * 72)
    print("  Treinamento concluído com sucesso.")
    print("  Arquivo gerado: diamantes_coulomb.png")
    print("  Relatório técnico finalizado.")
    print("=" * 72)

if __name__ == "__main__":
    main()
