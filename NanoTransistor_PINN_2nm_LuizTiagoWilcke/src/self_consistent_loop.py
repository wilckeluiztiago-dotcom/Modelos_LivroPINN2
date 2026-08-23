"""
Módulo: Loop Auto-Consistente Poisson-Schrödinger / Drift-Diffusion
Autor: Luiz Tiago Wilcke
"""

import torch
from equacao_poisson import ResidualPoisson
from perfil_dopagem_fosforo import PerfilDopagemFosforo


class LoopAutoConsistente:
    def __init__(self, modelo, perfil, mat, max_iter=20, tol=1e-5):
        self.modelo = modelo
        self.perfil = perfil
        self.mat = mat
        self.max_iter = max_iter
        self.tol = tol
        self.res_poisson = ResidualPoisson(mat)

    def passo(self, x_col):
        saida = self.modelo(x_col)
        phi = saida[:, 0:1]
        n = torch.abs(saida[:, 1:2]) + 1e-30
        Nd = self.perfil(x_col.detach())
        p = torch.zeros_like(n)
        Na = torch.zeros_like(n)
        res = self.res_poisson.residual(phi, n, p, Nd, Na, x_col)
        return torch.mean(res**2).item()

    def executar(self, x_col, otimizador=None):
        historico = []
        for it in range(self.max_iter):
            erro = self.passo(x_col)
            historico.append(erro)
            if erro < self.tol:
                print(f"Convergência atingida em {it+1} iterações (erro={erro:.2e})")
                break
            if otimizador is not None:
                otimizador.zero_grad()
                # re-computa loss e atualiza
                saida = self.modelo(x_col)
                phi = saida[:, 0:1]
                n = torch.abs(saida[:, 1:2]) + 1e-30
                Nd = self.perfil(x_col.detach())
                res = self.res_poisson.residual(phi, n, torch.zeros_like(n), Nd, torch.zeros_like(n), x_col)
                loss = torch.mean(res**2)
                loss.backward()
                otimizador.step()
        return historico
