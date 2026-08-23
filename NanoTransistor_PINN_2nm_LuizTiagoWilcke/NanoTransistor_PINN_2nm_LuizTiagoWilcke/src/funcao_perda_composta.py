"""
Módulo: Função de Perda Composta Multi-Física
Autor: Luiz Tiago Wilcke
"""

import torch
from equacao_poisson import ResidualPoisson
from equacao_schrodinger import ResidualSchrodinger
from continuidade_drift_diffusion import ResidualContinuidade
from condicoes_contorno import CondicoesContorno


class PerdaCompostaPINN:
    def __init__(self, mat=None, geo=None, Vds=0.5):
        self.res_poisson = ResidualPoisson(mat)
        self.res_schrodinger = ResidualSchrodinger(mat)
        self.res_continuidade = ResidualContinuidade(mat)
        self.bc = CondicoesContorno(Vds=Vds)
        # pesos adaptativos (podem ser treináveis)
        self.lambda_poisson = 1.0
        self.lambda_schrodinger = 0.5
        self.lambda_continuidade = 0.3
        self.lambda_bc = 10.0

    def __call__(self, modelo, x_col, y_col=None, Nd=None, perfil=None):
        """Calcula perda total a partir do modelo e pontos de colocação."""
        saida = modelo(x_col)
        phi = saida[:, 0:1]
        n = torch.abs(saida[:, 1:2]) + 1e-30   # densidade positiva
        # p ≈ 0 para dispositivo n-type
        p = torch.zeros_like(n)
        Na = torch.zeros_like(n)

        if Nd is None and perfil is not None:
            Nd = perfil(x_col.detach())
        if Nd is None:
            Nd = torch.ones_like(n) * 1e21

        # residual Poisson
        res_p = self.res_poisson.residual(phi, n, p, Nd, Na, x_col)
        loss_p = self.res_poisson.perda(res_p)

        # residual continuidade (estacionário)
        res_c = self.res_continuidade.residual_continuidade(n, phi, x_col)
        loss_c = self.res_continuidade.perda(res_c)

        # BC
        loss_bc = self.bc.perda_total_bc(modelo, n_pontos=64, device=x_col.device)

        loss = (self.lambda_poisson * loss_p +
                self.lambda_continuidade * loss_c +
                self.lambda_bc * loss_bc)
        return loss, {"poisson": loss_p.item(), "continuidade": loss_c.item(), "bc": loss_bc.item()}
