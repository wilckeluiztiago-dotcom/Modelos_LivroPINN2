"""
Módulo 01 - Fundamentos da Física de Reservatórios e Mecânica dos Meios Porosos
PINN para escoamento monofásico radial (equação de difusividade) em turbiditos da Margem Equatorial.
Autor: Luiz Tiago Wilcke
~200 linhas
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import sys
sys.path.append("..")
from utilitarios import RedePINN, residual_mse, gradiente, segunda_derivada, set_seed, device_disponivel
from dados_reais import DADOS_ME, converter_permeabilidade_mD_para_m2, obter_tensor

set_seed(42)
DEVICE = device_disponivel()


class PINNDarcyRadialME(nn.Module):
    """
    PINN para a equação de difusividade radial monofásica:
    (1/r) * d/dr (r * dP/dr) = (phi * mu * ct / k) * dP/dt
    Adaptada a reservatórios turbidíticos da Foz do Amazonas.
    """
    def __init__(self, camadas: list = [2, 64, 64, 64, 64, 1]):
        super().__init__()
        self.rede = RedePINN(camadas).to(DEVICE)
        self.k = torch.tensor(
            converter_permeabilidade_mD_para_m2(DADOS_ME["permeabilidade_media_mD"]),
            dtype=torch.float32, device=DEVICE
        )
        self.mu = torch.tensor(DADOS_ME["viscosidade_oleo_cP"] * 1e-3, device=DEVICE)  # Pa.s
        self.phi = obter_tensor("porosidade_media", DEVICE)
        self.ct = torch.tensor(
            DADOS_ME["compressibilidade_poros_1Pa"] + DADOS_ME["compressibilidade_fluido_1Pa"],
            device=DEVICE
        )
        self.rw = obter_tensor("raio_poco_m", DEVICE)
        self.re = obter_tensor("raio_drenagem_m", DEVICE)
        self.P_inicial = obter_tensor("pressao_inicial_bar", DEVICE) * 1e5  # Pa

    def forward(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Normalização simples para estabilidade
        r_norm = (r - self.rw) / (self.re - self.rw)
        t_norm = t / 86400.0  # dias
        x = torch.cat([r_norm, t_norm], dim=1)
        P_norm = self.rede(x)
        return P_norm * self.P_inicial  # escala de volta

    def residual_fisico(self, r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        r = r.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        P = self.forward(r, t)

        dP_dr = gradiente(P, r)
        d2P_dr2 = gradiente(dP_dr, r)
        dP_dt = gradiente(P, t)

        # Forma radial: (1/r) d/dr (r dP/dr) = (phi mu ct / k) dP/dt
        termo_espacial = (1.0 / (r + 1e-8)) * (dP_dr + r * d2P_dr2)
        coef = (self.phi * self.mu * self.ct) / self.k
        termo_temporal = coef * dP_dt
        return termo_espacial - termo_temporal

    def perda_total(
        self,
        r_col: torch.Tensor, t_col: torch.Tensor,
        r_bc: torch.Tensor, t_bc: torch.Tensor, P_bc: torch.Tensor,
        r_ic: torch.Tensor, t_ic: torch.Tensor, P_ic: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        res = self.residual_fisico(r_col, t_col)
        L_phys = residual_mse(res)

        P_pred_bc = self.forward(r_bc, t_bc)
        L_bc = residual_mse(P_pred_bc - P_bc)

        P_pred_ic = self.forward(r_ic, t_ic)
        L_ic = residual_mse(P_pred_ic - P_ic)

        total = L_phys + 20.0 * L_bc + 20.0 * L_ic
        return total, {"fisica": L_phys.item(), "contorno": L_bc.item(), "inicial": L_ic.item()}


def gerar_dados_treinamento(n_col: int = 2000, n_bc: int = 200, n_ic: int = 200):
    """Gera pontos de colocation, contorno e condição inicial."""
    rw = DADOS_ME["raio_poco_m"]
    re = DADOS_ME["raio_drenagem_m"]
    t_max = 30 * 86400.0  # 30 dias

    r_col = torch.rand(n_col, 1, device=DEVICE) * (re - rw) + rw
    t_col = torch.rand(n_col, 1, device=DEVICE) * t_max

    # Contorno no poço (Pwf constante exemplo)
    r_bc = torch.full((n_bc, 1), rw, device=DEVICE)
    t_bc = torch.rand(n_bc, 1, device=DEVICE) * t_max
    P_bc = torch.full((n_bc, 1), 300e5, device=DEVICE)  # 300 bar

    # Condição inicial
    r_ic = torch.rand(n_ic, 1, device=DEVICE) * (re - rw) + rw
    t_ic = torch.zeros(n_ic, 1, device=DEVICE)
    P_ic = torch.full((n_ic, 1), DADOS_ME["pressao_inicial_bar"] * 1e5, device=DEVICE)

    return r_col, t_col, r_bc, t_bc, P_bc, r_ic, t_ic, P_ic


def treinar(epocas: int = 3000, lr: float = 1e-3, print_a_cada: int = 500):
    modelo = PINNDarcyRadialME().to(DEVICE)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(otimizador, step_size=1000, gamma=0.5)

    r_col, t_col, r_bc, t_bc, P_bc, r_ic, t_ic, P_ic = gerar_dados_treinamento()

    historico = []
    for epoca in range(1, epocas + 1):
        otimizador.zero_grad()
        perda, detalhes = modelo.perda_total(r_col, t_col, r_bc, t_bc, P_bc, r_ic, t_ic, P_ic)
        perda.backward()
        otimizador.step()
        scheduler.step()

        if epoca % print_a_cada == 0 or epoca == 1:
            print(f"Época {epoca:5d} | Perda: {perda.item():.6e} | "
                  f"Física: {detalhes['fisica']:.3e} | BC: {detalhes['contorno']:.3e} | IC: {detalhes['inicial']:.3e}")
            historico.append((epoca, perda.item()))

    return modelo, historico


def predizer_perfil_pressao(modelo, t_dias: float = 10.0, n_pontos: int = 100):
    """Prediz perfil de pressão radial em um dado tempo."""
    rw = DADOS_ME["raio_poco_m"]
    re = DADOS_ME["raio_drenagem_m"]
    r = torch.linspace(rw, re, n_pontos, device=DEVICE).view(-1, 1)
    t = torch.full_like(r, t_dias * 86400.0)
    with torch.no_grad():
        P = modelo(r, t) / 1e5  # bar
    return r.cpu().numpy().flatten(), P.cpu().numpy().flatten()


if __name__ == "__main__":
    print("Treinando Módulo 01 - Fundamentos Petrofísica (Darcy Radial) - Margem Equatorial")
    print(f"Dispositivo: {DEVICE}")
    modelo, hist = treinar(epocas=2000, print_a_cada=400)
    r, P = predizer_perfil_pressao(modelo, t_dias=5.0)
    print(f"\nPerfil de pressão (t=5 dias): P(rw)={P[0]:.1f} bar, P(re)={P[-1]:.1f} bar")
    print("Módulo 01 concluído com sucesso.")
