"""
Módulo: Função de Perda Composta multi-física (pesos adaptativos)
Autor: Luiz Tiago Wilcke
"""

import torch
from equacao_poisson import ResidualPoisson
from continuidade_drift_diffusion import ResidualContinuidade
from condicoes_contorno import CondicoesContorno
from parametros_materiais_si import ParametrosSilicio


class PerdaCompostaPINN:
    def __init__(self, mat: ParametrosSilicio = None, L_nm: float = 14.0,
                 Vds: float = 0.5, T: float = 300.0):
        self.mat = mat or ParametrosSilicio()
        self.res_poisson = ResidualPoisson(self.mat, L_nm=L_nm, T=T)
        self.res_continuidade = ResidualContinuidade(self.mat, L_nm=L_nm, T=T)
        self.bc = CondicoesContorno(Vds=Vds, mat=self.mat, T=T)
        # pesos (podem ser adaptados durante o treino)
        self.lambda_p = 1.0
        self.lambda_c = 1.0
        self.lambda_bc = 20.0

    def __call__(self, modelo, x_col, perfil=None):
        saida = modelo(x_col)
        phi_star = saida[:, 0:1]
        # densidade normalizada positiva (softplus para estabilidade)
        n_star = torch.nn.functional.softplus(saida[:, 1:2]) + 1e-8
        p_star = torch.zeros_like(n_star)          # dispositivo n-type
        Na_star = torch.zeros_like(n_star)

        if perfil is not None:
            Nd_star = perfil.normalizado(x_col.detach())
        else:
            Nd_star = torch.ones_like(n_star)

        # residual Poisson
        res_p = self.res_poisson.residual(phi_star, n_star, p_star, Nd_star, Na_star, x_col)
        loss_p = self.res_poisson.perda(res_p)

        # residual continuidade
        res_c = self.res_continuidade.residual_continuidade(n_star, phi_star, x_col)
        loss_c = self.res_continuidade.perda(res_c)

        # condições de contorno
        loss_bc = self.bc.perda_total(modelo, perfil, n_pontos=96, device=x_col.device)

        loss = (self.lambda_p * loss_p +
                self.lambda_c * loss_c +
                self.lambda_bc * loss_bc)

        detalhes = {
            "poisson": loss_p.item(),
            "continuidade": loss_c.item(),
            "bc": loss_bc.item(),
            "total": loss.item()
        }
        return loss, detalhes
