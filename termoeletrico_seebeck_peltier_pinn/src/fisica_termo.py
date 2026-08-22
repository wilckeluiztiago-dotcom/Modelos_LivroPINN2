"""
Transporte termoelétrico não-linear: Seebeck + Peltier + Thomson.
Escala atômica / nanotransistor (fonte–canal–dreno).
"""

from typing import Dict


def parametros_termo_default() -> Dict[str, float]:
    return {
        "sigma": 1.0,      # condutividade elétrica
        "S": 0.15,         # coeficiente de Seebeck
        "kappa": 0.3,      # condutividade térmica
        "T_ref": 1.0,      # temperatura de referência
    }


# Relações constitutivas:
#   J  = -σ ∇φ - σ S ∇T
#   Π  = S T                    (Peltier)
#   q  = Π J - κ ∇T             (fluxo de calor)
#   ∇·q = J·E - ∇·(Π J)         (com E = -∇φ)
#       = J·E + Thomson + ...
