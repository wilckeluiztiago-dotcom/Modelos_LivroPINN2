"""
Módulo: Calibração Inversa de Dopagem / Work-function via PINN
Autor: Luiz Tiago Wilcke
"""

import torch
import torch.nn as nn
from perfil_dopagem_fosforo import PerfilDopagemFosforo


class CalibracaoInversa:
    """
    Ajusta parâmetros de dopagem (N_SD, N_canal, sigma) para
    minimizar diferença entre I-V prevista e dados-alvo.
    """
    def __init__(self, perfil: PerfilDopagemFosforo, modelo):
        self.perfil = perfil
        self.modelo = modelo
        self.otimizador = torch.optim.Adam(
            list(perfil.parameters()) + list(modelo.parameters()), lr=1e-3
        )

    def perda_dados(self, x, I_pred, I_alvo):
        return torch.mean((I_pred - I_alvo)**2)

    def passo(self, x_col, I_alvo):
        self.otimizador.zero_grad()
        saida = self.modelo(x_col)
        # proxy de corrente a partir de n e φ
        n = torch.abs(saida[:, 1:2])
        I_pred = n.mean() * 1e-3  # escala placeholder realista
        loss = self.perda_dados(x_col, I_pred, I_alvo)
        loss.backward()
        self.otimizador.step()
        return loss.item()
