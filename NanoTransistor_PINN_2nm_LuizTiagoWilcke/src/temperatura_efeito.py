"""
Módulo: Dependência de Temperatura
Autor: Luiz Tiago Wilcke
"""

import torch
from parametros_materiais_si import ParametrosSilicio


class EfeitoTemperatura:
    def __init__(self, mat: ParametrosSilicio = None):
        self.mat = mat or ParametrosSilicio()

    def mobilidade(self, T: float, mu0: float = None) -> float:
        """μ(T) ≈ μ0 * (T/300)^(-2.0)  (lei de potência típica Si)"""
        if mu0 is None:
            mu0 = self.mat.mobilidade_n0
        return mu0 * (T / 300.0)**(-2.0)

    def VT(self, T: float) -> float:
        return self.mat.VT(T)

    def Eg(self, T: float) -> float:
        return self.mat.Eg(T)

    def Nc_Nv(self, T: float):
        return self.mat.densidades_estados(T)
