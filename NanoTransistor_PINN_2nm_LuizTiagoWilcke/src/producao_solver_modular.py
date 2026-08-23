"""
Módulo: Solver Modular de Produção – interface completa e realista
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
from continuidade_drift_diffusion import ResidualContinuidade


class SolverPINNNanotransistor:
    def __init__(self, Vds: float = 0.5, T: float = 300.0, device: str = "cpu"):
        self.device = device
        self.Vds = Vds
        self.T = T
        self.geo = GeometriaNanotransistor()
        self.mat = ParametrosSilicio()
        self.perfil = PerfilDopagemFosforo(self.geo, self.mat, treinavel=False).to(device)
        self.modelo = PINNPoderosa(in_dim=1, out_dim=2, hidden=192, n_blocks=5).to(device)
        self.perda_fn = PerdaCompostaPINN(self.mat, L_nm=self.geo.comprimento_canal_nm,
                                          Vds=Vds, T=T)
        self.res_corrente = ResidualContinuidade(self.mat, L_nm=self.geo.comprimento_canal_nm, T=T)

    def treinar(self, epochs_adam: int = 1200, epochs_lbfgs: int = 50, n_col: int = 2000):
        x_col = amostragem_lhs(n_col, 1).to(self.device).requires_grad_(True)
        print(f"Rede: {self.modelo.num_parametros():,} parâmetros | "
              f"Colocação: {n_col} pontos | Vds={self.Vds} V")
        hist_a = treinar_adam(self.modelo, self.perda_fn, x_col,
                              epochs=epochs_adam, perfil=self.perfil, device=self.device)
        hist_l = treinar_lbfgs(self.modelo, self.perda_fn, x_col,
                               max_iter=epochs_lbfgs, perfil=self.perfil)
        return hist_a + hist_l

    def prever(self, x: torch.Tensor):
        self.modelo.eval()
        with torch.no_grad():
            return self.modelo(x)

    def potencial_V(self, x: torch.Tensor) -> torch.Tensor:
        """Potencial em Volts."""
        saida = self.prever(x)
        return saida[:, 0:1] * self.mat.VT(self.T)

    def densidade_m3(self, x: torch.Tensor) -> torch.Tensor:
        """Densidade de elétrons em m⁻³."""
        saida = self.prever(x)
        n_star = torch.nn.functional.softplus(saida[:, 1:2]) + 1e-8
        return n_star * self.mat.N_D_SD

    def corrente_aproximada(self, x: torch.Tensor) -> torch.Tensor:
        """Corrente aproximada (A/m) a partir do residual de continuidade."""
        x = x.clone().requires_grad_(True)
        saida = self.modelo(x)
        phi_star = saida[:, 0:1]
        n_star = torch.nn.functional.softplus(saida[:, 1:2]) + 1e-8
        J_star = self.res_corrente.corrente_eletrons(n_star, phi_star, x)
        return self.res_corrente.corrente_fisica_A_por_m(J_star)

    def salvar(self, caminho: str):
        torch.save({
            "modelo": self.modelo.state_dict(),
            "perfil": self.perfil.state_dict(),
            "Vds": self.Vds,
            "T": self.T,
        }, caminho)

    def carregar(self, caminho: str):
        ckpt = torch.load(caminho, map_location=self.device)
        self.modelo.load_state_dict(ckpt["modelo"])
        self.perfil.load_state_dict(ckpt["perfil"])
