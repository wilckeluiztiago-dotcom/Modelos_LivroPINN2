"""
Módulo 02 - Modelagem Dinâmica de Poços Verticais e Escoamento Vertical Multifásico
Gradiente de pressão vertical (gravitacional + friccional + acelerativo) para poços
em águas ultraprofundas da Margem Equatorial.
Autor: Luiz Tiago Wilcke
~200 linhas
"""

import torch
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..")
from utilitarios import RedePINN, residual_mse, gradiente, set_seed, device_disponivel
from dados_reais import DADOS_ME

set_seed(42)
DEVICE = device_disponivel()


class PINNGradientePressaoVertical(nn.Module):
    """
    PINN para o gradiente de pressão vertical em poço:
    dP/dz = componente_gravitacional + componente_friccional + componente_acelerativo
    Adaptado a lâmina d'água ~2900 m (Morpho).
    """
    def __init__(self, camadas=[2, 48, 48, 48, 1]):
        super().__init__()
        self.rede = RedePINN(camadas).to(DEVICE)
        self.g = 9.81
        self.rho_oleo = DADOS_ME["densidade_oleo_kgm3"]
        self.D = 0.1  # diâmetro interno tubulação (m) exemplo
        self.rugosidade = 4.5e-5

    def forward(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """z: profundidade (m), q: vazão (m³/s) → P (Pa)"""
        z_norm = z / 4000.0
        q_norm = q / 0.1
        x = torch.cat([z_norm, q_norm], dim=1)
        P_norm = self.rede(x)
        return P_norm * 500e5  # escala ~500 bar

    def residual_gradiente(self, z: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        z = z.clone().requires_grad_(True)
        q = q.clone().requires_grad_(True)
        P = self.forward(z, q)
        dP_dz = gradiente(P, z)

        # Componentes simplificados (Beggs-Brill / Hagedorn-Brown adaptados)
        # Gravitacional (holdup médio aproximado)
        holdup = 0.6  # fração líquida média
        dp_grav = holdup * self.rho_oleo * self.g

        # Friccional (Darcy-Weisbach simplificado)
        v = q / (np.pi * (self.D/2)**2 + 1e-12)
        Re = self.rho_oleo * v * self.D / 0.001  # viscosidade aproximada
        f = 0.046 * (Re ** -0.2)  # Blasius
        dp_fric = f * (self.rho_oleo * v**2) / (2 * self.D)

        # Acelerativo (desprezado em regime permanente ou pequeno)
        dp_acel = 0.0

        dp_total = dp_grav + dp_fric + dp_acel
        return dP_dz - dp_total

    def perda(self, z_col, q_col, z_bc, q_bc, P_bc):
        res = self.residual_gradiente(z_col, q_col)
        L_phys = residual_mse(res)
        P_pred = self.forward(z_bc, q_bc)
        L_bc = residual_mse(P_pred - P_bc)
        return L_phys + 15.0 * L_bc, {"fisica": L_phys.item(), "contorno": L_bc.item()}


def gerar_pontos(n_col=1500, n_bc=150):
    z_max = DADOS_ME["lamina_agua_m"] + DADOS_ME["profundidade_vertical_m"]
    z_col = torch.rand(n_col, 1, device=DEVICE) * z_max
    q_col = torch.rand(n_col, 1, device=DEVICE) * 0.05 + 0.01  # 0.01-0.06 m³/s

    z_bc = torch.zeros(n_bc, 1, device=DEVICE)  # superfície
    q_bc = torch.rand(n_bc, 1, device=DEVICE) * 0.05 + 0.01
    P_bc = torch.full((n_bc, 1), 20e5, device=DEVICE)  # 20 bar na cabeça

    return z_col, q_col, z_bc, q_bc, P_bc


def treinar(epocas=2500, lr=1e-3):
    modelo = PINNGradientePressaoVertical().to(DEVICE)
    opt = torch.optim.Adam(modelo.parameters(), lr=lr)
    z_col, q_col, z_bc, q_bc, P_bc = gerar_pontos()

    for epoca in range(1, epocas + 1):
        opt.zero_grad()
        perda, det = modelo.perda(z_col, q_col, z_bc, q_bc, P_bc)
        perda.backward()
        opt.step()
        if epoca % 500 == 0:
            print(f"Época {epoca:5d} | Perda: {perda.item():.4e} | Física: {det['fisica']:.3e}")
    return modelo


if __name__ == "__main__":
    print("Treinando Módulo 02 - Gradiente de Pressão Vertical (Margem Equatorial)")
    modelo = treinar()
    print("Módulo 02 concluído.")
