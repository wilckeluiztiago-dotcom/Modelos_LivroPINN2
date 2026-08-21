"""
CFET quântico: nFET empilhado sobre pFET com barreiras sub-nm.
Acoplamento eletrostático e de troca de spin entre canais.
"""

import numpy as np
from typing import Optional, Tuple, Dict
from .ising_glauber import RedeIsingGlauber
from .langevin_potencial import ProcessoLangevin, passo_langevin


class CFETSpinLangevin:
    """
    Complementary-FET com dois canais empilhados:

        ┌─────────────┐
        │   nFET      │  ← spins Ising_n + potencial Langevin φ_n
        ├─────────────┤  barreira dielétrica sub-nm
        │   pFET      │  ← spins Ising_p + potencial Langevin φ_p
        └─────────────┘

    A ocupação de sub-banda é discreta (Ising/Glauber);
    o potencial de canal evolui continuamente (Langevin)
    acoplado à magnetização do outro canal (troca + eletrostática).
    """

    def __init__(
        self,
        n_spins: int = 24,
        J_intra: float = 1.0,          # troca dentro do canal
        J_inter: float = 0.35,         # troca entre n e p (barreira sub-nm)
        kappa_ep: float = 0.4,         # acoplamento eletrostático φ ↔ m
        beta: float = 1.2,
        gamma: float = 1.0,
        sigma: float = 0.12,
        semente: Optional[int] = 42,
    ):
        self.J_inter = J_inter
        self.kappa_ep = kappa_ep
        g = np.random.default_rng(semente)

        # redes de Ising por canal
        self.ising_n = RedeIsingGlauber(
            n_spins=n_spins, J_troca=J_intra, h_campo=0.1, beta=beta, semente=semente
        )
        self.ising_p = RedeIsingGlauber(
            n_spins=n_spins, J_troca=J_intra, h_campo=-0.1, beta=beta, semente=semente + 1
        )

        # potenciais de canal (Langevin)
        self.langevin_n = ProcessoLangevin(
            phi0=0.2, gamma=gamma, sigma=sigma, a=1.0, b=0.25, c=0.05, semente=semente + 2
        )
        self.langevin_p = ProcessoLangevin(
            phi0=-0.2, gamma=gamma, sigma=sigma, a=1.0, b=0.25, c=-0.05, semente=semente + 3
        )

        self.n_spins = n_spins

    def passo_acoplado(self, dt: float = 0.01, n_glauber: int = 5) -> Dict[str, float]:
        """
        Um passo do sistema acoplado:
        1) atualiza campos de Ising com acoplamento inter-canal
        2) vários passos Glauber
        3) Langevin de φ_n e φ_p com força de magnetização cruzada
        """
        m_n = self.ising_n.magnetizacao()
        m_p = self.ising_p.magnetizacao()

        # campo efetivo: gate local + troca inter-canal + potencial do outro canal
        self.ising_n.definir_campo(0.1 + self.J_inter * m_p + self.kappa_ep * self.langevin_p.phi)
        self.ising_p.definir_campo(-0.1 + self.J_inter * m_n + self.kappa_ep * self.langevin_n.phi)

        for _ in range(n_glauber):
            self.ising_n.passo_glauber()
            self.ising_p.passo_glauber()

        m_n = self.ising_n.magnetizacao()
        m_p = self.ising_p.magnetizacao()

        # Langevin acoplado: φ sente a magnetização do canal oposto
        self.langevin_n.passo(dt, acoplamento_spin=self.kappa_ep, m=m_p)
        self.langevin_p.passo(dt, acoplamento_spin=self.kappa_ep, m=m_n)

        return {
            "m_n": m_n,
            "m_p": m_p,
            "phi_n": self.langevin_n.phi,
            "phi_p": self.langevin_p.phi,
        }

    def simular(self, n_passos: int = 2000, dt: float = 0.01) -> Dict[str, np.ndarray]:
        hist = {
            "m_n": np.zeros(n_passos),
            "m_p": np.zeros(n_passos),
            "phi_n": np.zeros(n_passos),
            "phi_p": np.zeros(n_passos),
            "t": np.arange(n_passos) * dt,
        }
        for k in range(n_passos):
            s = self.passo_acoplado(dt=dt)
            hist["m_n"][k] = s["m_n"]
            hist["m_p"][k] = s["m_p"]
            hist["phi_n"][k] = s["phi_n"]
            hist["phi_p"][k] = s["phi_p"]
        return hist
