"""
Módulo: Residual da Equação de Poisson (unidades normalizadas)
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class ResidualPoisson:
    """
    Residual da equação de Poisson em 1D normalizado.
    Trabalha com potencial em Volts e densidades normalizadas para estabilidade numérica.
    """
    def __init__(self, mat: ParametrosSilicio = None, L_nm: float = 14.0):
        self.mat = mat or ParametrosSilicio()
        self.eps = self.mat.epsilon_si()
        self.q = self.mat.q
        self.L = L_nm * 1e-9  # metros

    def residual(self, phi: torch.Tensor, n: torch.Tensor, p: torch.Tensor,
                 Nd: torch.Tensor, Na: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forma residual: d²φ/dx² + (q/ε)(p - n + Nd - Na) = 0
        x é normalizado [0,1]; escala espacial é absorvida.
        """
        # gradientes via Autograd
        dphi_dx = torch.autograd.grad(
            phi, x, grad_outputs=torch.ones_like(phi),
            create_graph=True, retain_graph=True
        )[0]
        d2phi_dx2 = torch.autograd.grad(
            dphi_dx, x, grad_outputs=torch.ones_like(dphi_dx),
            create_graph=True, retain_graph=True
        )[0]

        # fator de escala: (L² * q / ε) * densidades
        # usamos densidades em m⁻³ e escalamos residual
        escala = (self.L**2 * self.q) / self.eps
        rho_norm = escala * (p - n + Nd - Na)
        residual = d2phi_dx2 + rho_norm
        return residual

    def perda(self, residual: torch.Tensor) -> torch.Tensor:
        return torch.mean(residual**2)

    def perda_ponderada(self, residual: torch.Tensor, pesos: torch.Tensor = None) -> torch.Tensor:
        if pesos is None:
            return self.perda(residual)
        return torch.mean(pesos * residual**2)


if __name__ == "__main__":
    print("ResidualPoisson carregado - Luiz Tiago Wilcke")
