"""
Módulo: Solver Modular de Produção (template final)
Autor: Luiz Tiago Wilcke
"""

import torch
from arquitetura_pinn_poderosa import PINNPoderosa
from perfil_dopagem_fosforo import PerfilDopagemFosforo
from funcao_perda_composta import PerdaCompostaPINN
from otimizacao_hibrida import treinar_adam, treinar_lbfgs
from amostragem_lhs import amostragem_lhs
from geometria_dispositivo import GeometriaNanotransistor
from parametros_materiais_si import ParametrosSilicio


class SolverPINNNanotransistor:
    """
    Interface de alto nível para o solver completo.
    """
    def __init__(self, device="cpu"):
        self.device = device
        self.geo = GeometriaNanotransistor()
        self.mat = ParametrosSilicio()
        self.perfil = PerfilDopagemFosforo(self.geo, self.mat).to(device)
        self.modelo = PINNPoderosa(in_dim=1, out_dim=2, hidden=128, n_blocks=4).to(device)
        self.perda_fn = PerdaCompostaPINN(self.mat, self.geo, Vds=0.5)

    def treinar(self, epochs_adam=1500, epochs_lbfgs=50, n_col=2000):
        x_col = amostragem_lhs(n_col, 1).to(self.device).requires_grad_(True)
        hist_adam = treinar_adam(
            self.modelo, self.perda_fn, x_col,
            epochs=epochs_adam, perfil=self.perfil, device=self.device
        )
        hist_lbfgs = treinar_lbfgs(
            self.modelo, self.perda_fn, x_col,
            max_iter=epochs_lbfgs, perfil=self.perfil
        )
        return hist_adam + hist_lbfgs

    def prever(self, x):
        self.modelo.eval()
        with torch.no_grad():
            return self.modelo(x)

    def salvar(self, caminho="modelo_nanotransistor.pt"):
        torch.save({
            "modelo": self.modelo.state_dict(),
            "perfil": self.perfil.state_dict(),
        }, caminho)

    def carregar(self, caminho):
        ckpt = torch.load(caminho, map_location=self.device)
        self.modelo.load_state_dict(ckpt["modelo"])
        self.perfil.load_state_dict(ckpt["perfil"])
