"""
Módulo: Condições de Contorno (Dirichlet, Neumann, abertas)
Autor: Luiz Tiago Wilcke
"""

import torch


class CondicoesContorno:
    def __init__(self, Vds: float = 0.5, Vgs: float = 0.7, phi_fonte: float = 0.0):
        self.Vds = Vds
        self.Vgs = Vgs
        self.phi_fonte = phi_fonte

    def perda_dirichlet_fonte(self, modelo, n_pontos=100, device="cpu"):
        x = torch.zeros(n_pontos, 1, device=device, requires_grad=True)
        phi = modelo(x)[:, 0:1]
        return torch.mean((phi - self.phi_fonte)**2)

    def perda_dirichlet_dreno(self, modelo, n_pontos=100, device="cpu"):
        x = torch.ones(n_pontos, 1, device=device, requires_grad=True)
        phi = modelo(x)[:, 0:1]
        return torch.mean((phi - self.Vds)**2)

    def perda_total_bc(self, modelo, n_pontos=100, device="cpu"):
        return (self.perda_dirichlet_fonte(modelo, n_pontos, device) +
                self.perda_dirichlet_dreno(modelo, n_pontos, device))
