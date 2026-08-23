"""
Módulo: Parâmetros Materiais do Silício – constantes físicas rigorosas
Autor: Luiz Tiago Wilcke
"""

from dataclasses import dataclass
import math


@dataclass
class ParametrosSilicio:
    # --- Constantes fundamentais (SI) ---
    q: float = 1.602176634e-19       # C
    k_B: float = 1.380649e-23        # J/K
    hbar: float = 1.054571817e-34    # J·s
    h: float = 6.62607015e-34        # J·s
    m0: float = 9.1093837015e-31     # kg
    epsilon0: float = 8.8541878128e-12  # F/m

    # --- Silício @ 300 K ---
    epsilon_r_si: float = 11.7
    m_star_trans: float = 0.258      # elétrons transversais <100>
    m_star_long: float = 0.916
    m_star_dos: float = 1.08         # densidade de estados efetiva
    Eg_300K: float = 1.124           # eV (precise)
    afinidade: float = 4.05          # eV
    Nc_300K: float = 2.86e25         # m^-3
    Nv_300K: float = 3.10e25         # m^-3  (valor atualizado)
    ni_300K: float = 1.07e16         # m^-3  (intrínseca Si)
    mobilidade_n0: float = 0.1400    # m²/V·s  (bulk baixa campo)
    mobilidade_p0: float = 0.0450
    vsat_n: float = 1.0e5            # m/s
    vsat_p: float = 8.0e4
    tau_n_srh: float = 1.0e-6        # s (melhor qualidade)
    tau_p_srh: float = 1.0e-6
    C_n_auger: float = 2.8e-31       # m⁶/s
    C_p_auger: float = 9.9e-32

    # --- Dopagem típica nó N2 (fósforo) ---
    N_D_SD: float = 2.0e26           # 2e20 cm^-3
    N_D_canal: float = 1.0e21        # 1e15 cm^-3
    N_A_substrato: float = 1.0e21

    def epsilon_si(self) -> float:
        return self.epsilon_r_si * self.epsilon0

    def VT(self, T: float = 300.0) -> float:
        """Tensão térmica kT/q (V)."""
        return (self.k_B * T) / self.q

    def Eg(self, T: float = 300.0) -> float:
        """Gap Varshni (eV)."""
        return 1.166 - 4.73e-4 * T**2 / (T + 636.0)

    def densidades_estados(self, T: float = 300.0):
        Nc = self.Nc_300K * (T / 300.0)**1.5
        Nv = self.Nv_300K * (T / 300.0)**1.5
        return Nc, Nv

    def ni(self, T: float = 300.0) -> float:
        """Concentração intrínseca (m^-3)."""
        Nc, Nv = self.densidades_estados(T)
        Eg_J = self.Eg(T) * self.q
        return math.sqrt(Nc * Nv) * math.exp(-Eg_J / (2 * self.k_B * T))

    def massa_efetiva_kg(self) -> float:
        return self.m_star_trans * self.m0

    def comprimento_debye(self, N: float, T: float = 300.0) -> float:
        """Comprimento de Debye (m) para normalização."""
        return math.sqrt(self.epsilon_si() * self.VT(T) / (self.q * max(N, 1e18)))

    def resumo(self) -> str:
        return (f"Si: εr={self.epsilon_r_si}, m*={self.m_star_trans}m0, "
                f"Eg={self.Eg_300K:.3f} eV, ni={self.ni_300K:.2e} m⁻³, "
                f"μn={self.mobilidade_n0} m²/Vs")
