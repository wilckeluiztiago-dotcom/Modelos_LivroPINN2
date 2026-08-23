"""Módulo: Solver Auto-Consistente Poisson-NEGF 1D para Nanotransistores

Autor: Luiz Tiago Wilcke
Descrição: Código completo com modelo Tight-Binding, Funções de Green Não-Equilíbrio (NEGF),
           equação de Poisson com acoplamento capacitivo de porta e loop auto-consistente (SCF).
"""

from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class NEGF1DSolver:
  """Solver quântico baseado no formalismo de Funções de Green Não-Equilíbrio."""

  Q_E = 1.602176634e-19  # C (Carga do elétron)
  H_PLANCK = 6.62607015e-34  # J·s (Constante de Planck)
  H_BAR = 1.054571817e-34  # J·s
  M_E = 9.1093837e-31  # kg (Massa do elétron livre)
  K_B_EV = 8.617333262e-5  # eV/K (Constante de Boltzmann)
  G0 = 2 * (1.602176634e-19**2) / 6.62607015e-34  # 2q²/h ≈ 77.48 µS

  def __init__(
      self,
      N_pts: int = 60,
      a_dx: float = 0.5e-9,  # Espaçamento de grade (m) -> 0.5 nm
      m_eff: float = 0.25,  # Massa efetiva (m* / m0)
      gamma_S: float = 0.15,  # Acoplamento Fonte (eV)
      gamma_D: float = 0.15,  # Acoplamento Dreno (eV)
      E_F: float = 0.0,  # Nível de Fermi de equilíbrio (eV)
      T: float = 300.0,  # Temperatura (K)
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_pts
    self.dx = a_dx
    self.device = device
    self.gamma_S = gamma_S
    self.gamma_D = gamma_D
    self.E_F = E_F
    self.T = T
    self.kBT = max(self.K_B_EV * self.T, 1e-6)

    # Parâmetro de hopping t0 = hbar^2 / (2 * m* * dx^2) em eV
    t0_joules = (self.H_BAR**2) / (2.0 * (m_eff * self.M_E) * (self.dx**2))
    self.t0 = float(t0_joules / self.Q_E)  # Conversão para eV

    # Hamiltoniano cinético base H0 (Tridiagonal)
    diag_kinetic = 2.0 * self.t0 * torch.ones(self.N, dtype=torch.float64)
    off_kinetic = -self.t0 * torch.ones(self.N - 1, dtype=torch.float64)
    self.H0 = (
        torch.diag(diag_kinetic)
        + torch.diag(off_kinetic, 1)
        + torch.diag(off_kinetic, -1)
    ).to(self.device)

    # Matrizes de autoenergia estáticas (Wide-Band Approximation)
    self.Sigma_S = torch.zeros(
        (self.N, self.N), dtype=torch.complex128, device=self.device
    )
    self.Sigma_D = torch.zeros(
        (self.N, self.N), dtype=torch.complex128, device=self.device
    )
    self.Sigma_S[0, 0] = -1j * (self.gamma_S / 2.0)
    self.Sigma_D[-1, -1] = -1j * (self.gamma_D / 2.0)

    self.Gamma_S = 1j * (self.Sigma_S - self.Sigma_S.conj().T)
    self.Gamma_D = 1j * (self.Sigma_D - self.Sigma_D.conj().T)
    self.I_mat = torch.eye(
        self.N, dtype=torch.complex128, device=self.device
    )

  def fermi_dirac(
      self, E: torch.Tensor, mu: float
  ) -> torch.Tensor:
    """Ocupação de Fermi-Dirac protegida contra underflow/overflow."""
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def resolver_densidade_e_transmissao(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula a densidade linear de portadores n(x) [1/m] e transmissão T(E)."""
    # H = H0 + diag(U(x))
    H_total = self.H0 + torch.diag(U_potencial.to(torch.float64))
    H_c = H_total.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    T_lista = []
    n_x = torch.zeros(self.N, dtype=torch.float64, device=self.device)

    # Varredura em energia
    for E in E_grid:
      E_val = E.item()
      f_S = self.fermi_dirac(E, mu_S)
      f_D = self.fermi_dirac(E, mu_D)

      # G^R(E) = [E*I - H - Sigma_S - Sigma_D]^(-1)
      A_mat = (E_val + 1e-7j) * self.I_mat - H_c - self.Sigma_S - self.Sigma_D
      G_R = torch.linalg.inv(A_mat)
      G_A = G_R.conj().T

      # T(E) = Tr[Gamma_S * G_R * Gamma_D * G_A]
      T_E = torch.trace(self.Gamma_S @ G_R @ self.Gamma_D @ G_A).real
      T_lista.append(T_E)

      # G^<(E) = G_R * (Gamma_S*f_S + Gamma_D*f_D) * G_A
      Sigma_in = (self.Gamma_S * f_S) + (self.Gamma_D * f_D)
      G_lesser = G_R @ Sigma_in @ G_A

      # Densidade local por intervalo de energia: diag(G^<) / (2*pi*dx)
      n_x += torch.diag(G_lesser).imag / (2.0 * np.pi * self.dx)

    dE = (E_grid[1] - E_grid[0]).item()
    n_x = n_x * dE
    T_espectro = torch.stack(T_lista)

    return n_x, T_espectro

  def calcular_corrente(
      self, T_espectro: torch.Tensor, Vds: float, E_grid: torch.Tensor
  ) -> float:
    """Calcula a corrente terminal via integral de Landauer-Büttiker (A)."""
    mu_S = self.E_F
    mu_D = self.E_F - Vds
    f_S = self.fermi_dirac(E_grid, mu_S)
    f_D = self.fermi_dirac(E_grid, mu_D)

    integrando = T_espectro * (f_S - f_D)
    integral_T = torch.trapezoid(integrando, E_grid).item()
    return float(self.G0 * integral_T)


class Poisson1D:
  """Solver da Equação de Poisson 1D com acoplamento capacitivo de porta (DG-MOSFET)."""

  EPS_0 = 8.8541878128e-12  # F/m
  EPS_SEMI = 11.7 * 8.8541878128e-12  # Silício
  Q_E = 1.602176634e-19

  def __init__(
      self,
      N_pts: int,
      dx: float,
      lambda_g: float = 3.0e-9,
      device: torch.device = torch.device("cpu"),
  ):
    """lambda_g: Comprimento de triagem característico da porta (m)."""
    self.N = N_pts
    self.dx = dx
    self.lambda_sq = lambda_g**2
    self.device = device

    # Construção da matriz Laplaciana 1D com termo de porta: d²phi/dx² - phi/lambda²
    diag_val = -2.0 / (dx**2) - (1.0 / self.lambda_sq)
    off_val = 1.0 / (dx**2)

    self.M_poisson = (
        torch.diag(
            torch.full((self.N,), diag_val, dtype=torch.float64)
        )
        + torch.diag(
            torch.full(
                (self.N - 1,), off_val, dtype=torch.float64
            ),
            1,
        )
        + torch.diag(
            torch.full(
                (self.N - 1,), off_val, dtype=torch.float64
            ),
            -1,
        )
    ).to(self.device)

  def resolver_potencial(
      self,
      n_x: torch.Tensor,
      N_dop: torch.Tensor,
      V_gate: float,
      V_source: float,
      V_drain: float,
  ) -> torch.Tensor:
    """Resolve: d²phi/dx² - (phi - V_gate)/lambda² = -q * (N_dop - n_x) / eps_semi."""
    # Vetor de carga do lado direito
    rho = self.Q_E * (N_dop - n_x)
    RHS = -(rho / self.EPS_SEMI) - (V_gate / self.lambda_sq)

    # Condições de Contorno de Dirichlet nos contatos
    A = self.M_poisson.clone()
    b = RHS.clone()

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = V_source

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = V_drain

    phi = torch.linalg.solve(A, b)
    return phi


class AutoConsistentePoissonNEGF:
  """Loop Auto-Consistente (Self-Consistent Field - SCF) com amortecimento de Anderson/Picard."""

  def __init__(
      self,
      negf_solver: NEGF1DSolver,
      poisson_solver: Poisson1D,
      max_iter: int = 60,
      tolerancia: float = 1e-4,
      alpha_mix: float = 0.15,
  ):
    self.negf = negf_solver
    self.poisson = poisson_solver
    self.max_iter = max_iter
    self.tol = tolerancia
    self.alpha = alpha_mix

  def executar(
      self,
      V_gate: float,
      V_ds: float,
      N_dop: torch.Tensor,
      E_grid: torch.Tensor,
      phi_inicial: torch.Tensor = None,
  ) -> Dict[str, torch.Tensor]:
    """Executa as iterações de ponto fixo entre Poisson e NEGF."""
    if phi_inicial is None:
      # Chute inicial linear
      phi = torch.linspace(
          0.0, V_ds, self.negf.N, dtype=torch.float64, device=self.negf.device
      )
    else:
      phi = phi_inicial.clone()

    n_x = torch.zeros(
        self.negf.N, dtype=torch.float64, device=self.negf.device
    )

    for i in range(self.max_iter):
      # Energia potencial eletrostática U(x) = -q * phi(x) em eV
      U_potencial = -phi

      # 1. Passo Quântico (NEGF)
      n_x_novo, T_E = self.negf.resolver_densidade_e_transmissao(
          U_potencial, V_ds, E_grid
      )

      # 2. Passo Eletrostático (Poisson)
      phi_novo = self.poisson.resolver_potencial(
          n_x_novo, N_dop, V_gate, 0.0, V_ds
      )

      # 3. Avaliação de Erro e Atualização com Mistura (Picard damping)
      erro = torch.max(torch.abs(phi_novo - phi)).item()
      phi = (1.0 - self.alpha) * phi + self.alpha * phi_novo
      n_x = (1.0 - self.alpha) * n_x + self.alpha * n_x_novo

      if erro < self.tol:
        break

    corrente = self.negf.calcular_corrente(T_E, V_ds, E_grid)

    return {
        "potencial_phi": phi,
        "energia_U": -phi,
        "densidade_n": n_x,
        "transmissao": T_E,
        "corrente_A": corrente,
        "convergido": erro < self.tol,
        "iteracoes": i + 1,
    }


if __name__ == "__main__":
  # Configuração da Simulação
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  N_pontos = 50
  dx_passo = 0.4e-9  # 0.4 nm por sítio -> Canal de 20 nm
  L_total = N_pontos * dx_passo * 1e9  # nm

  # Perfil de Dopagem (N+ Fonte e Dreno, Canal Intrínseco N+/i/N+)
  N_doping = torch.zeros(N_pontos, dtype=torch.float64, device=dispositivo)
  N_doping[:10] = 5e25  # Fonte N+ (1/m³)
  N_doping[-10:] = 5e25  # Dreno N+ (1/m³)
  N_doping[10:-10] = 1e20  # Canal Intrínseco

  # Instanciação dos Solvers
  solver_negf = NEGF1DSolver(
      N_pts=N_pontos,
      a_dx=dx_passo,
      m_eff=0.20,
      gamma_S=0.2,
      gamma_D=0.2,
      E_F=0.1,
      T=300.0,
      device=dispositivo,
  )
  solver_poisson = Poisson1D(
      N_pts=N_pontos, dx=dx_passo, lambda_g=2.5e-9, device=dispositivo
  )
  loop_scf = AutoConsistentePoissonNEGF(
      solver_negf, solver_poisson, max_iter=80, tolerancia=5e-5, alpha_mix=0.2
  )

  # Grid de Energia para integração
  grid_E = torch.linspace(-0.5, 1.5, 400, dtype=torch.float64, device=dispositivo)

  # Simulação sob diferentes tensões de Porta (Vgs)
  V_ds_teste = 0.35  # V
  portas_Vgs = [0.0, 0.3, 0.6]
  resultados = []

  print(
      f"--- Iniciando Solver Poisson-NEGF 1D (Canal: {L_total:.1f} nm, Vds ="
      f" {V_ds_teste} V) ---"
  )
  for Vg in portas_Vgs:
    res = loop_scf.executar(
        V_gate=Vg, V_ds=V_ds_teste, N_dop=N_doping, E_grid=grid_E
    )
    resultados.append(res)
    print(
        f"Vgs = {Vg:.2f} V | Convergiu: {res['convergido']} em"
        f" {res['iteracoes']} iterações | I_ds = {res['corrente_A'] * 1e6:.3f}"
        " µA"
    )

  # Exibição dos Perfis Físicos Obtidos
  x_nm = np.linspace(0, L_total, N_pontos)

  plt.figure(figsize=(12, 5))

  plt.subplot(1, 2, 1)
  for i, Vg in enumerate(portas_Vgs):
    plt.plot(
        x_nm,
        resultados[i]["energia_U"].cpu().numpy(),
        label=f"Vgs = {Vg:.1f} V",
    )
  plt.title("Perfil de Banda / Potencial U(x)")
  plt.xlabel("Posição x (nm)")
  plt.ylabel("Energia Potencial (eV)")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend()

  plt.subplot(1, 2, 2)
  for i, Vg in enumerate(portas_Vgs):
    plt.semilogy(
        x_nm,
        resultados[i]["densidade_n"].cpu().numpy() + 1e15,
        label=f"Vgs = {Vg:.1f} V",
    )
  plt.title("Densidade Eletrônica n(x)")
  plt.xlabel("Posição x (nm)")
  plt.ylabel("Densidade Linear (m⁻¹)")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend()

  plt.tight_layout()
  plt.show()
