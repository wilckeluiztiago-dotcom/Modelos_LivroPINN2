"""
Módulo: Condições de Contorno realistas (contatos ôhmicos)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class CondicoesContorno:
    """
    Contatos ôhmicos:
      φ(0) = 0          (fonte, referência)
      φ(1) = Vds / VT   (dreno, normalizado)
      n(0) ≈ Nd(0)/N_ref
      n(1) ≈ Nd(1)/N_ref
    """
    def __init__(self, Vds: float = 0.5, mat: ParametrosSilicio = None, T: float = 300.0):
        self.Vds = Vds
        self.mat = mat or ParametrosSilicio()
        self.VT = self.mat.VT(T)
        self.phi_dreno_star = Vds / self.VT

    def perda_potencial(self, modelo, n_pontos: int = 128, device="cpu"):
        x_s = torch.zeros(n_pontos, 1, device=device, requires_grad=True)
        x_d = torch.ones(n_pontos, 1, device=device, requires_grad=True)
        phi_s = modelo(x_s)[:, 0:1]
        phi_d = modelo(x_d)[:, 0:1]
        loss_s = torch.mean(phi_s**2)
        loss_d = torch.mean((phi_d - self.phi_dreno_star)**2)
        return loss_s + loss_d

    def perda_densidade(self, modelo, perfil, n_pontos: int = 128, device="cpu"):
        x_s = torch.zeros(n_pontos, 1, device=device)
        x_d = torch.ones(n_pontos, 1, device=device)
        n_s = torch.abs(modelo(x_s)[:, 1:2])
        n_d = torch.abs(modelo(x_d)[:, 1:2])
        Nd_s = perfil.normalizado(x_s)
        Nd_d = perfil.normalizado(x_d)
        return torch.mean((n_s - Nd_s)**2) + torch.mean((n_d - Nd_d)**2)

    def perda_total(self, modelo, perfil=None, n_pontos: int = 128, device="cpu"):
        loss = self.perda_potencial(modelo, n_pontos, device)
        if perfil is not None:
            loss = loss + 0.5 * self.perda_densidade(modelo, perfil, n_pontos, device)
        return loss
