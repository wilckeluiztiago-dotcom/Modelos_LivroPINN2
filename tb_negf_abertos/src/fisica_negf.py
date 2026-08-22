"""
TB-NEGF atomístico: canal Si:P acoplado a leads n+ semi-infinitos.
"""

from typing import Dict, Tuple
import numpy as np


def parametros_negf_default() -> Dict:
    return {
        "n_sites": 7,          # sítios do canal (ímpar → P no centro)
        "t_hop": 1.0,          # hopping NN
        "E_onsite": 0.0,       # onsite Si
        "V_P": -2.5,           # potencial do doador no centro
        "t_lead": 1.0,         # hopping no lead
        "E_lead": 0.0,         # onsite lead
        "eta": 1e-3,           # iη para G^R
        "mu_L": 0.3,           # potencial químico lead esquerdo
        "mu_R": -0.3,          # lead direito
        "kT": 0.05,
    }


def hamiltoniano_canal(p: Dict) -> np.ndarray:
    """
    Cadeia 1D TB com doador P no centro.
    H_{i,i}=E0, H_{i,i+1}=-t; sítio central += V_P.
    """
    n = p["n_sites"]
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        H[i, i] = p["E_onsite"]
        if i < n - 1:
            H[i, i + 1] = -p["t_hop"]
            H[i + 1, i] = -p["t_hop"]
    centro = n // 2
    H[centro, centro] += p["V_P"]
    return H


def autoenergia_lead_1d(E: complex, t_lead: float = 1.0, E_lead: float = 0.0) -> complex:
    """
    Autoenergia de lead 1D semi-infinito (solução analítica):
      Σ = (E - E_lead)/2 - sign * sqrt( ((E-E_lead)/2)^2 - t^2 )
    ramo com Im(Σ) ≤ 0 para G^R.
    """
    x = (E - E_lead) / 2.0
    disc = x ** 2 - t_lead ** 2
    # raiz com parte imaginária negativa (ou zero)
    sq = np.sqrt(disc + 0j)
    if sq.imag > 0:
        sq = -sq
    # se real e |x|<|t|, escolher ramo atrasado
    Sigma = x - sq
    # garantir Im Σ ≤ 0
    if Sigma.imag > 0:
        Sigma = x + sq
    return complex(Sigma)


def Sigma_contatos(E: float, n: int, p: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Σ_L e Σ_R acoplados apenas às pontas do canal."""
    eta = p["eta"]
    Ec = E + 1j * eta
    sig = autoenergia_lead_1d(Ec, p["t_lead"], p["E_lead"])
    # acoplamento lead-canal ≈ t_hop
    t_c = p["t_hop"]
    Sigma_L = np.zeros((n, n), dtype=complex)
    Sigma_R = np.zeros((n, n), dtype=complex)
    Sigma_L[0, 0] = (t_c ** 2) * sig / (p["t_lead"] ** 2 + 1e-15) * p["t_lead"]
    # forma padrão: Σ = τ g_lead τ† ≈ t² * σ_lead / estrutura
    # simplificação wide-band-like com forma 1D:
    Sigma_L[0, 0] = t_c * sig * t_c / max(abs(p["t_lead"]), 1e-12) * 0.5 + t_c * sig
    # mais limpo: Σ_{00} = t_c² / t_lead² * Σ_surface, use Σ_surface = sig
    Sigma_L[0, 0] = (t_c ** 2 / max(p["t_lead"], 1e-12)) * (sig / max(abs(p["t_lead"]), 1e-12)) * p["t_lead"]
    # Standard: for 1D chain with hop t, surface GF leads to Σ = t * σ where σ is reduced.
    # Use: Σ_L[0,0] = t_hop * sig  (sig already has energy units of hop)
    Sigma_L = np.zeros((n, n), dtype=complex)
    Sigma_R = np.zeros((n, n), dtype=complex)
    Sigma_L[0, 0] = (t_c ** 2) * (sig / p["t_lead"] if abs(p["t_lead"]) > 1e-12 else sig)
    Sigma_R[n - 1, n - 1] = (t_c ** 2) * (sig / p["t_lead"] if abs(p["t_lead"]) > 1e-12 else sig)
    # normalize: common textbook Σ = t * e^{ik} style — use direct
    Sigma_L[0, 0] = sig * (t_c / p["t_lead"]) ** 2 * p["t_lead"] if p["t_lead"] else sig
    # Final clean form used in many codes:
    Sigma_L = np.zeros((n, n), dtype=complex)
    Sigma_R = np.zeros((n, n), dtype=complex)
    # Σ = t² g_s ; g_s = Σ_1D / t²  →  Σ_contact = sig (when t_lead=t_hop)
    Sigma_L[0, 0] = sig
    Sigma_R[n - 1, n - 1] = sig
    return Sigma_L, Sigma_R


def green_retardada(E: float, H: np.ndarray, p: Dict) -> np.ndarray:
    n = H.shape[0]
    Sigma_L, Sigma_R = Sigma_contatos(E, n, p)
    A = (E + 1j * p["eta"]) * np.eye(n) - H - Sigma_L - Sigma_R
    return np.linalg.inv(A)


def Gamma_de_Sigma(Sigma: np.ndarray) -> np.ndarray:
    return 1j * (Sigma - Sigma.conj().T)


def transmissao(E: float, H: np.ndarray, p: Dict) -> float:
    n = H.shape[0]
    Sigma_L, Sigma_R = Sigma_contatos(E, n, p)
    GR = green_retardada(E, H, p)
    GA = GR.conj().T
    GL = Gamma_de_Sigma(Sigma_L)
    GR_mat = Gamma_de_Sigma(Sigma_R)
    T = np.real(np.trace(GL @ GR @ GR_mat @ GA))
    return float(max(T, 0.0))


def fermi(E: np.ndarray, mu: float, kT: float) -> np.ndarray:
    x = (E - mu) / max(kT, 1e-8)
    return 1.0 / (1.0 + np.exp(np.clip(x, -40, 40)))


def corrente_landauer(
    H: np.ndarray,
    p: Dict,
    n_E: int = 200,
    E_min: float = -4.0,
    E_max: float = 4.0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    E = np.linspace(E_min, E_max, n_E)
    T = np.array([transmissao(e, H, p) for e in E])
    dE = E[1] - E[0]
    fL = fermi(E, p["mu_L"], p["kT"])
    fR = fermi(E, p["mu_R"], p["kT"])
    I = float(np.sum(T * (fL - fR)) * dE)  # unidades 2e/h = 1
    return I, E, T
