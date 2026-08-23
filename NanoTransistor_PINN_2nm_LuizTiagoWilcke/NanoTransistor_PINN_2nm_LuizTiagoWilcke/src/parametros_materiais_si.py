"""
Módulo: Parâmetros Materiais do Silício e Constantes Físicas
Autor: Luiz Tiago Wilcke
"""

import torch
from dataclasses import dataclass


@dataclass
class ParametrosSilicio:
    # Constantes físicas fundamentais
    q: float = 1.60217662e-19          # C
    k_B: float = 1.380649e-23          # J/K
    hbar: float = 1.0545718e-34        # J·s
    m0: float = 9.1093837e-31          # kg
    epsilon0: float = 8.854187817e-12  # F/m
    h: float = 6.62607015e-34          # J·s

    # Silício (300 K)
    epsilon_r_si: float = 11.7
    m_star_trans: float = 0.26         # m*/m0 transversal <100>
    m_star_long: float = 0.98
    Eg_300K: float = 1.12              # eV
    afinidade: float = 4.05            # eV
    Nc_300K: float = 2.8e25            # m^-3
    Nv_300K: float = 1.04e25           # m^-3
    mobilidade_n0: float = 0.1400      # m²/V·s (baixa campo)
    mobilidade_p0: float = 0.0480
    tau_n_srh: float = 1e-7            # s
    tau_p_srh: float = 1e-7

    # Dopagem típica N2 (fósforo)
    N_D_SD: float = 2.0e26             # m^-3 = 2e20 cm^-3
    N_D_canal: float = 1.0e21          # m^-3 = 1e15 cm^-3
    N_A_substrato: float = 1.0e21

    def epsilon_si(self) -> float:
        return self.epsilon_r_si * self.epsilon0

    def VT(self, T: float = 300.0) -> float:
        """Tensão térmica kT/q em Volts."""
        return (self.k_B * T) / self.q

    def densidades_estados(self, T: float = 300.0):
        Nc = self.Nc_300K * (T / 300.0)**1.5
        Nv = self.Nv_300K * (T / 300.0)**1.5
        return Nc, Nv

    def massa_efetiva_kg(self) -> float:
        return self.m_star_trans * self.m0

    def Eg(self, T: float = 300.0) -> float:
        """Gap de energia com dependência fraca de T (Varshni simplificado)."""
        return self.Eg_300K - 4.73e-4 * T**2 / (T + 636)

    def resumo(self) -> str:
        return (f"Si: εr={self.epsilon_r_si}, m*={self.m_star_trans}m0, "
                f"Eg={self.Eg_300K} eV, μn={self.mobilidade_n0} m²/Vs")


if __name__ == "__main__":
    mat = ParametrosSilicio()
    print(mat.resumo())
    print(f"VT(300K) = {mat.VT():.4f} V")
    print(f"ε_Si = {mat.epsilon_si():.3e} F/m")
