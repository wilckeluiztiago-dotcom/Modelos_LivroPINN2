"""
Eletromigração atômica — equação de Korhonen modificada.
Interconexões Ru/Mo em nós de ~1 nm, J > 10^7 A/cm².
"""

from typing import Dict


def parametros_korhonen_default() -> Dict[str, float]:
    """
    Parâmetros efetivos normalizados (unidades reduzidas).
    D_a, B, Ω, Z*, ρ combinados em D_eff e fator de vento eletrônico.
    """
    return {
        "D_eff": 0.15,       # D_a B Ω / (k_B T) efetivo
        "Z_star_e_Omega": 2.0,  # (Z* e / Ω) efetivo no gradiente de φ
        "sigma_cond": 1.0,   # condutividade elétrica
        "L": 1.0,            # comprimento da linha
    }


# Equação de Korhonen:
#   ∂σ_H/∂t = ∇ · [ D_eff ( ∇σ_H − (Z* e ρ / Ω) J ) ]
# com J = −σ_cond ∇φ,  ∇·(σ_cond ∇φ)=0
#
# Em 1D, J ≈ constante (contorno isolante de fluxo de massa nas pontas
# com bloqueio de blocking boundary).
