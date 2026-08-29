"""
Módulo 04 - Escoamento em Meios Porosos Anisotrópicos e Inversão de Permeabilidade
PINN para identificação inversa do tensor de permeabilidade + regularização TV.
Autor: Luiz Tiago Wilcke
~200 linhas
"""

import torch
import torch.nn as nn
import sys
sys.path.append("..")
from utilitarios import RedePINN, residual_mse, gradiente, set_seed, device_disponivel
from dados_reais import DADOS_ME, converter_permeabilidade_mD_para_m2

set_seed(42)
DEVICE = device_disponivel()


class PINNInversaoPermeabilidade(nn.Module):
    """
    Problema inverso: observar pressão → inferir campo de permeabilidade k(x,y).
    Regularização de Variação Total (TV) para preservar descontinuidades geológicas.
    """
    def __init__(self):
        super().__init__()
        # Rede para pressão
        self.rede_P = RedePINN([2, 64, 64, 64, 1]).to(DEVICE)
        # Rede para log-permeabilidade (garante positividade)
        self.rede_logk = RedePINN([2, 48, 48, 48, 1]).to(DEVICE)
        self.mu = torch.tensor(0.001, device=DEVICE)
        self.phi = torch.tensor(DADOS_ME["porosidade_media"], device=DEVICE)
        self.ct = torch.tensor(2.5e-9, device=DEVICE)

    def pressao(self, x, y):
        entrada = torch.cat([x, y], dim=1)
        return self.rede_P(entrada)

    def permeabilidade(self, x, y):
        entrada = torch.cat([x, y], dim=1)
        logk = self.rede_logk(entrada)
        k0 = converter_permeabilidade_mD_para_m2(DADOS_ME["permeabilidade_media_mD"])
        return k0 * torch.exp(logk)  # k > 0

    def residual_darcy(self, x, y):
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        P = self.pressao(x, y)
        k = self.permeabilidade(x, y)

        dP_dx = gradiente(P, x)
        dP_dy = gradiente(P, y)

        # Divergência de (k/μ ∇P) ≈ 0 (estacionário, sem fonte)
        flux_x = (k / self.mu) * dP_dx
        flux_y = (k / self.mu) * dP_dy
        dflux_x = gradiente(flux_x, x)
        dflux_y = gradiente(flux_y, y)
        return dflux_x + dflux_y

    def regularizacao_tv(self, x, y):
        """Variação Total do campo de log-k."""
        x = x.clone().requires_grad_(True)
        y = y.clone().requires_grad_(True)
        logk = self.rede_logk(torch.cat([x, y], dim=1))
        dlogk_dx = gradiente(logk, x)
        dlogk_dy = gradiente(logk, y)
        return torch.mean(torch.sqrt(dlogk_dx**2 + dlogk_dy**2 + 1e-8))

    def perda(self, x_col, y_col, x_obs, y_obs, P_obs, lambda_tv=0.01):
        res = self.residual_darcy(x_col, y_col)
        L_phys = residual_mse(res)
        P_pred = self.pressao(x_obs, y_obs)
        L_data = residual_mse(P_pred - P_obs)
        L_tv = self.regularizacao_tv(x_col, y_col)
        total = L_phys + 50.0 * L_data + lambda_tv * L_tv
        return total, {"fisica": L_phys.item(), "dados": L_data.item(), "tv": L_tv.item()}


def treinar_inversao(epocas=2000):
    modelo = PINNInversaoPermeabilidade()
    opt = torch.optim.Adam(modelo.parameters(), lr=5e-4)

    # Dados sintéticos de observação (simulando sensores)
    n_obs = 100
    x_obs = torch.rand(n_obs, 1, device=DEVICE)
    y_obs = torch.rand(n_obs, 1, device=DEVICE)
    P_obs = 400e5 - 50e5 * (x_obs + y_obs)  # gradiente artificial

    x_col = torch.rand(1500, 1, device=DEVICE)
    y_col = torch.rand(1500, 1, device=DEVICE)

    for epoca in range(1, epocas + 1):
        opt.zero_grad()
        perda, det = modelo.perda(x_col, y_col, x_obs, y_obs, P_obs)
        perda.backward()
        opt.step()
        if epoca % 400 == 0:
            print(f"Época {epoca:5d} | Total: {perda.item():.4e} | "
                  f"Física: {det['fisica']:.3e} | Dados: {det['dados']:.3e} | TV: {det['tv']:.3e}")
    return modelo


if __name__ == "__main__":
    print("Treinando Módulo 04 - Inversão de Permeabilidade Anisotrópica")
    modelo = treinar_inversao()
    print("Módulo 04 concluído.")
