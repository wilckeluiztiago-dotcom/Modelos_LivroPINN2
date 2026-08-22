"""
Kadanoff–Baym em tempo real (nível único / orbital efetivo).
Auto-energia HF + kernel de memória tipo GW simplificado.
"""

from typing import Dict
import numpy as np


def parametros_kb_default() -> Dict[str, float]:
    return {
        "hbar": 1.0,
        "eps0": 0.5,         # energia do nível
        "U_HF": 0.3,         # Hartree–Fock efetivo
        "gamma": 0.4,        # acoplamento ao banho / lead
        "n_eq": 0.5,         # ocupação de equilíbrio
        "tau_mem": 0.6,      # tempo de memória
        "t_max": 4.0,
    }


def Sigma_HF(n: float, p: Dict) -> float:
    """Σ_HF = U ⟨n⟩"""
    return p["U_HF"] * n


def kernel_memoria(dt: np.ndarray, p: Dict) -> np.ndarray:
    """K(τ) ∝ exp(−|τ|/τ_mem) e^{−i ε τ}  (GW-like local)"""
    tau = p["tau_mem"]
    return (p["gamma"] / tau) * np.exp(-np.abs(dt) / tau) * np.exp(-1j * p["eps0"] * dt)
