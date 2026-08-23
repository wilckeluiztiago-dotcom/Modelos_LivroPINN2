"""
Módulo 02: Parâmetros Materiais do Silício e Dopagem de Fósforo
Autor: Luiz Tiago Wilcke
"""

import torch
from dataclasses import dataclass

@dataclass
class ParametrosSilicio:
    # Constantes físicas
    q: float = 1.60217662e-19          # carga elementar (C)
    k_B: float = 1.380649e-23          # Boltzmann (J/K)
    hbar: float = 1.0545718e-34        # ħ (J·s)
    m0: float = 9.1093837e-31          # massa do elétron livre
    epsilon0: float = 8.854187817e-12  # vácuo

    # Silício
    epsilon_r_si: float = 11.7
    m_star_transversal: float = 0.26   # m*/m0 para elétrons em <100>
    m_star_longitudinal: float = 0.98
    Eg_300K: float = 1.12              # eV
    afinidade: float = 4.05            # eV
    Nc_300K: float = 2.8e25            # m^-3
    Nv_300K: float = 1.04e25           # m^-3
    mobilidade_eletrons: float = 0.14  # m²/V·s (baixa campo, bulk)
    mobilidade_buracos: float = 0.048

    # Dopagem típica N2 (fósforo)
    N_D_SD: float = 2.0e26             # m^-3 ≡ 2e20 cm^-3
    N_D_canal: float = 1.0e21          # m^-3 ≡ 1e15 cm^-3
    N_A_substrato: float = 1.0e21

    def epsilon_si(self):
        return self.epsilon_r_si * self.epsilon0

    def VT(self, T: float = 300.0):
        """Tensão térmica kT/q em Volts."""
        return (self.k_B * T) / self.q

    def densidades_estados(self, T: float = 300.0):
        """Nc, Nv escalados com T."""
        Nc = self.Nc_300K * (T / 300.0)**1.5
        Nv = self.Nv_300K * (T / 300.0)**1.5
        return Nc, Nv

if __name__ == "__main__":
    mat = ParametrosSilicio()
    print(f"ε_Si = {mat.epsilon_si():.3e} F/m")
    print(f"VT(300K) = {mat.VT():.4f} V")
    print(f"N_D S/D = {mat.N_D_SD:.2e} m⁻³")
