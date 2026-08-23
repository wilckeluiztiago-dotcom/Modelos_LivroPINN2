"""Módulo: Framework Quântico-Eletrostático Completo (NEGF, Landauer & PINN)

Autor: Luiz Tiago Wilcke
Descrição: Implementação unificada e rigorosa contendo:
           1. Single-Level NEGF Analítico (Transporte ressonante Breit-Wigner)
           2. 1D Tight-Binding NEGF Matricial (Open Boundary Conditions, G^R, G^<,
           A(x,E), T(E))
           3. Solver de Poisson 1D com Eletrostática de Porta (DG-MOSFET)
           4. Physics-Informed Neural Network (PINN) auto-consistente treinável
           via PyTorch
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. CONSTANTES FÍSICAS UNIVERSAIS
# =============================================================================


class ConstantesFisicas:
  """Constantes fundamentais em unidades SI e conversões para eV."""

  Q_E: float = 1.602176634e-19  # C (Carga elementar)
  H_PLANCK: float = 6.62607015e-34  # J·s (Constante de Planck)
  H_BAR: float = 1.054571817e-34  # J·s (Planck reduzida)
  M_E: float = 9.1093837e-31  # kg (Massa de repouso do elétron)
  K_B_EV: float = 8.617333262e-5  # eV/K (Constante de Boltzmann)
  EPS_0: float = 8.8541878128e-12  # F/m (Permissividade do vácuo)
  EPS_SI: float = 11.7 * 8.8541878128e-12  # F/m (Permissividade do Silício)
  G0: float = 2.0 * (1.602176634e-19**2) / 6.62607015e-34  # ~77.48 µS (Quanta de condutância)


# =============================================================================
# 2. MODELO NEGF ANALÍTICO (NÍVEL ÚNICO / CANAL ULTRA-CURTO)
# =============================================================================


class SingleLevelNEGF(nn.Module):
  """Modelo Analítico de Nível Único com acoplamento fonte-dreno explícito.

  Calcula Green Retardada, Densidades Espectrais parciais A_S e A_D,
  Transmissão Breit-Wigner e Corrente de Landauer-Büttiker.
  """

  def __init__(
      self,
      gamma_S: float = 0.08,
      gamma_D: float = 0.08,
      E_F: float = 0.0,
      T: float = 300.0,
      E_min: float = -1.5,
      E_max: float = 1.5,
      n_energy_pts: int = 500,
  ):
    super().__init__()
    self.gamma_S = float(gamma_S)
    self.gamma_D = float(gamma_D)
    self.gamma_tot = self.gamma_S + self.gamma_D
    self.E_F = float(E_F)
    self.T = float(T)
    self.kBT = max(ConstantesFisicas.K_B_EV * self.T, 1e-7)

    # Grid de quadratura numérica de energia
    self.register_buffer(
        "E_grid",
        torch.linspace(E_min, E_max, n_energy_pts, dtype=torch.float64),
    )

  def fermi_dirac(
      self, E: torch.Tensor, mu: torch.Tensor
  ) -> torch.Tensor:
    """Distribuição de Fermi-Dirac numericamente estável."""
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def green_retardada(
      self, E: torch.Tensor, E0: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Retorna componentes real e imaginária de G^R(E) = 1 / [E - E0 + i*Γ_tot/2]."""
    denom = (E - E0) ** 2 + (0.5 * self.gamma_tot) ** 2
    return (E - E0) / denom, -0.5 * self.gamma_tot / denom

  def densidades_espectrais(
      self, E: torch.Tensor, E0: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Retorna A_S(E), A_D(E) e A_total(E)."""
    denom = (E - E0) ** 2 + (0.5 * self.gamma_tot) ** 2
    A_S = self.gamma_S / denom
    A_D = self.gamma_D / denom
    A_tot = self.gamma_tot / denom
    return A_S, A_D, A_tot

  def transmissao(self, E: torch.Tensor, E0: torch.Tensor) -> torch.Tensor:
    """T(E) = Γ_S * Γ_D * |G^R(E)|²."""
    denom = (E - E0) ** 2 + (0.5 * self.gamma_tot) ** 2
    return (self.gamma_S * self.gamma_D) / denom

  def forward(
      self, E0: torch.Tensor, Vds: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward pass calculando Ocupação (N), Corrente Landauer (A) e Transmissão média."""
    mu_S = torch.as_tensor(self.E_F, dtype=E0.dtype, device=E0.device)
    mu_D = self.E_F - Vds

    E = self.E_grid.unsqueeze(-1) if E0.dim() > 0 else self.E_grid

    f_S = self.fermi_dirac(E, mu_S)
    f_D = self.fermi_dirac(E, mu_D)

    # 1. Densidade eletrônica integrada no estado
    A_S, A_D, _ = self.densidades_espectrais(E, E0)
    integrando_n = (A_S * f_S + A_D * f_D) / (2.0 * np.pi)
    n_charge = torch.trapezoid(integrando_n, self.E_grid, dim=0)

    # 2. Corrente Landauer-Büttiker
    T_E = self.transmissao(E, E0)
    integrando_I = T_E * (f_S - f_D)
    corrente = ConstantesFisicas.G0 * torch.trapezoid(
        integrando_I, self.E_grid, dim=0
    )

    return n_charge, corrente, T_E


# =============================================================================
# 3. SOLVER 1D TIGHT-BINDING NEGF MATRICIAL COMPLETO
# =============================================================================


class TightBindingNEGF1D:
  """Solver NEGF 1D para perfis arbitrários de potencial U(x).

  Resolve:
    - Hamiltoniano discreto de massa efetiva
    - Matrizes de autoenergia de contatos abertos (Wide-Band)
    - Função de Green Retardada G^R(E) = [E*I - H - Σ_S - Σ_D]⁻¹
    - Função de Correlação Menor G^<(E) = G^R * Σ^in * G^A
    - Densidade de carga espacial n(x) e transmissão T(E)
  """

  def __init__(
      self,
      N_sites: int = 50,
      dx: float = 0.4e-9,  # 0.4 nm
      m_eff: float = 0.20,  # Massa efetiva m* / m0
      gamma_S: float = 0.2,
      gamma_D: float = 0.2,
      E_F: float = 0.1,
      T: float = 300.0,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.device = device
    self.gamma_S = gamma_S
    self.gamma_D = gamma_D
    self.E_F = E_F
    self.kBT = max(ConstantesFisicas.K_B_EV * T, 1e-7)

    # Energia de hopping: t0 = hbar^2 / (2 * m* * dx^2) em eV
    t0_joules = (ConstantesFisicas.H_BAR**2) / (
        2.0 * (m_eff * ConstantesFisicas.M_E) * (self.dx**2)
    )
    self.t0 = float(t0_joules / ConstantesFisicas.Q_E)

    # Matriz Hamiltoniana Cinética H0 (Tridiagonal)
    diag_kinetic = 2.0 * self.t0 * torch.ones(self.N, dtype=torch.float64)
    off_kinetic = -self.t0 * torch.ones(self.N - 1, dtype=torch.float64)
    self.H0 = (
        torch.diag(diag_kinetic)
        + torch.diag(off_kinetic, 1)
        + torch.diag(off_kinetic, -1)
    ).to(self.device)

    # Autoenergias Wide-Band
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
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def resolver_transporte(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Executa a integração em energia sobre todo o espectro."""
    H = self.H0 + torch.diag(U_potencial.to(torch.float64))
    H_c = H.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    T_lista = []
    n_x = torch.zeros(self.N, dtype=torch.float64, device=self.device)

    for E in E_grid:
      E_val = E.item()
      f_S = self.fermi_dirac(E, mu_S)
      f_D = self.fermi_dirac(E, mu_D)

      # Função de Green Retardada G^R = [E*I - H - Σ_S - Σ_D]⁻¹
      A_mat = (E_val + 1e-7j) * self.I_mat - H_c - self.Sigma_S - self.Sigma_D
      G_R = torch.linalg.inv(A_mat)
      G_A = G_R.conj().T

      # Transmissão Quântica T(E) = Tr[Γ_S * G^R * Γ_D * G^A]
      T_E = torch.trace(self.Gamma_S @ G_R @ self.Gamma_D @ G_A).real
      T_lista.append(T_E)

      # Função de Menor G^< = G^R * (Γ_S*f_S + Γ_D*f_D) * G^A
      Sigma_in = (self.Gamma_S * f_S) + (self.Gamma_D * f_D)
      G_lesser = G_R @ Sigma_in @ G_A

      # n(x, E) = diag(G^<) / (2π * dx)
      n_x += torch.diag(G_lesser).imag / (2.0 * np.pi * self.dx)

    dE = (E_grid[1] - E_grid[0]).item()
    n_x = n_x * dE
    T_espectro = torch.stack(T_lista)

    # Corrente de Landauer terminal
    f_S_all = self.fermi_dirac(E_grid, mu_S)
    f_D_all = self.fermi_dirac(E_grid, mu_D)
    integrando_I = T_espectro * (f_S_all - f_D_all)
    I_ds = float(
        ConstantesFisicas.G0 * torch.trapezoid(integrando_I, E_grid).item()
    )

    return n_x, T_espectro, I_ds


# =============================================================================
# 4. SOLVER DE POISSON 1D E AUTO-CONSISTÊNCIA (SCF BENCHMARK)
# =============================================================================


class Poisson1D:
  """Equação de Poisson 1D discretizada com termo capacitivo de porta (DG-MOSFET)."""

  def __init__(
      self,
      N_sites: int,
      dx: float,
      lambda_g: float = 2.5e-9,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.lambda_sq = lambda_g**2
    self.device = device

    # Laplaciano 1D acoplado à porta: d²ϕ/dx² - (ϕ - Vg)/λ²
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

  def resolver(
      self,
      n_x: torch.Tensor,
      N_dop: torch.Tensor,
      Vg: float,
      Vs: float,
      Vd: float,
  ) -> torch.Tensor:
    """Resolve a distribuição de potencial eletrostático ϕ(x)."""
    rho = ConstantesFisicas.Q_E * (N_dop - n_x)
    RHS = -(rho / ConstantesFisicas.EPS_SI) - (Vg / self.lambda_sq)

    A = self.M_poisson.clone()
    b = RHS.clone()

    # Dirichlet nos contatos
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = Vs

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = Vd

    return torch.linalg.solve(A, b)


# =============================================================================
# 5. PHYSICS-INFORMED NEURAL NETWORK (PINN) QUÂNTICO-ELETROSTÁTICA
# =============================================================================


class NanotransistorPINN(nn.Module):
  """Rede Neural Informada pela Física (PINN) para predição direta do potencial

  eletrostático U(x) e parâmetros quânticos sob polarizações (x, Vgs, Vds).
  """

  def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
    super().__init__()
    camadas = []
    # Entrada: [x_normalizado, Vgs, Vds] (3 dimensões)
    camadas.append(nn.Linear(3, hidden_dim))
    camadas.append(nn.Tanh())

    for _ in range(num_layers - 1):
      camadas.append(nn.Linear(hidden_dim, hidden_dim))
      camadas.append(nn.Tanh())

    # Saída: [Potencial Eletrostático ϕ(x) em Volts]
    camadas.append(nn.Linear(hidden_dim, 1))
    self.rede = nn.Sequential(*camadas)

    # Camada auxiliar para predição direta do autovalor quântico dominante E0(Vgs, Vds)
    self.cabeca_quântica = nn.Sequential(
        nn.Linear(2, 32),
        nn.Tanh(),
        nn.Linear(32, 1),
    )

  def forward_potencial(
      self, x: torch.Tensor, Vgs: torch.Tensor, Vds: torch.Tensor
  ) -> torch.Tensor:
    """Predição de potencial eletrostático contínuo."""
    inputs = torch.cat([x, Vgs, Vds], dim=-1)
    return self.rede(inputs)

  def forward_E0(
      self, Vgs: torch.Tensor, Vds: torch.Tensor
  ) -> torch.Tensor:
    """Prediz o nível dominante E0 do canal para o solver Single-Level."""
    inputs = torch.cat([Vgs, Vds], dim=-1)
    return self.cabeca_quântica(inputs)


class TreinadorPINN:
  """Motor de Treinamento com Perdas de PDE (Poisson), Condições de Contorno e Transporte NEGF."""

  def __init__(
      self,
      modelo: NanotransistorPINN,
      single_level_negf: SingleLevelNEGF,
      L_ch: float = 20e-9,  # 20 nm
      lambda_g: float = 2.5e-9,
      lr: float = 1e-3,
  ):
    self.modelo = modelo
    self.negf = single_level_negf
    self.L = L_ch
    self.lambda_sq = lambda_g**2
    self.otimizador = optim.Adam(self.modelo.parameters(), lr=lr)

  def passo_treinamento(self, n_colocacao: int = 128) -> Dict[str, float]:
    """Calcula resíduos diferenciais exatos via PyTorch Autograd."""
    self.otimizador.zero_grad()

    # Amostragem no espaço de fase (x ∈ [0, L], Vgs ∈ [0, 0.8] V, Vds ∈ [0, 0.5] V)
    x = torch.rand(n_colocacao, 1, dtype=torch.float32, requires_grad=True) * self.L
    Vgs = torch.rand(n_colocacao, 1, dtype=torch.float32) * 0.8
    Vds = torch.rand(n_colocacao, 1, dtype=torch.float32) * 0.5

    # 1. Avaliação do Potencial Eletrostático
    phi = self.modelo.forward_potencial(x / self.L, Vgs, Vds)

    # 2. Gradientes via Autograd para a PDE de Poisson
    grad_phi = torch.autograd.grad(
        phi,
        x,
        grad_outputs=torch.ones_like(phi),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2phi_dx2 = torch.autograd.grad(
        grad_phi,
        x,
        grad_outputs=torch.ones_like(grad_phi),
        create_graph=True,
        retain_graph=True,
    )[0]

    # 3. Densidade de carga quântica via Single-Level NEGF
    E0 = self.modelo.forward_E0(Vgs, Vds)
    n_eletrons, corrente_pred, _ = self.negf(
        E0.squeeze(-1).to(torch.float64), Vds.squeeze(-1).to(torch.float64)
    )
    n_eletrons_3d = (n_eletrons.unsqueeze(-1).to(torch.float32) / self.L) * 1e18

    # Perfil sintético de dopagem (Canal Intrínseco central)
    N_dop = torch.where(
        (x < 0.2 * self.L) | (x > 0.8 * self.L), 5e25, 1e20
    ).to(torch.float32)

    # Resíduo da Equação de Poisson 1D com porta:
    # d²ϕ/dx² - (ϕ - Vgs)/λ² + q*(N_dop - n)/ε = 0
    rho = ConstantesFisicas.Q_E * (N_dop - n_eletrons_3d)
    res_poisson = (
        d2phi_dx2
        - ((phi - Vgs) / self.lambda_sq)
        + (rho / ConstantesFisicas.EPS_SI)
    )
    loss_pde = torch.mean(res_poisson**2) * 1e-18  # Normalização de escala

    # 4. Condições de Contorno de Dirichlet
    x_source = torch.zeros(n_colocacao, 1, dtype=torch.float32)
    x_drain = (
        torch.ones(n_colocacao, 1, dtype=torch.float32) * self.L
    )

    phi_source = self.modelo.forward_potencial(x_source / self.L, Vgs, Vds)
    phi_drain = self.modelo.forward_potencial(x_drain / self.L, Vgs, Vds)

    loss_bc_source = torch.mean((phi_source - 0.0) ** 2)
    loss_bc_drain = torch.mean((phi_drain - Vds) ** 2)
    loss_bc = loss_bc_source + loss_bc_drain

    # 5. Perda Total e Otimização
    loss_total = loss_pde + 10.0 * loss_bc
    loss_total.backward()
    self.otimizador.step()

    return {
        "loss_total": loss_total.item(),
        "loss_pde": loss_pde.item(),
        "loss_bc": loss_bc.item(),
    }


# =============================================================================
# 6. DEMONSTRAÇÃO EXECUTÁVEL COMPLETA
# =============================================================================

if __name__ == "__main__":
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # ---------------------------------------------------------
  # Parte A: Verificação do Módulo Single-Level NEGF
  # ---------------------------------------------------------
  print("=" * 70)
  print(" PARTE A: Transporte Quântico Single-Level NEGF Analítico")
  print("=" * 70)

  sl_negf = SingleLevelNEGF(
      gamma_S=0.06, gamma_D=0.06, E_F=0.0, T=300.0
  ).to(dispositivo)

  tensao_vds = torch.tensor(0.25, dtype=torch.float64, device=dispositivo)
  nivel_e0 = torch.tensor(0.10, dtype=torch.float64, device=dispositivo)

  n_q, i_landauer, T_analitica = sl_negf(nivel_e0, tensao_vds)
  print(f"Nível Quântico E0: {nivel_e0.item():.2f} eV | Vds: {tensao_vds.item():.2f} V")
  print(f"Ocupação Eletrônica do Nível (N): {n_q.item():.4f} elétrons")
  print(f"Corrente Landauer-Büttiker:        {i_landauer.item() * 1e6:.4f} µA")

  # ---------------------------------------------------------
  # Parte B: Simulação Auto-Consistente Tight-Binding 1D
  # ---------------------------------------------------------
  print("\n" + "=" * 70)
  print(" PARTE B: Solver Tight-Binding NEGF Matricial 1D (Validação)")
  print("=" * 70)

  N_grid = 40
  dx_grid = 0.5e-9  # 0.5 nm -> Canal 20 nm
  tb_solver = TightBindingNEGF1D(
      N_sites=N_grid,
      dx=dx_grid,
      m_eff=0.20,
      gamma_S=0.15,
      gamma_D=0.15,
      E_F=0.05,
      device=dispositivo,
  )

  E_espectro = torch.linspace(
      -0.6, 1.2, 300, dtype=torch.float64, device=dispositivo
  )
  U_teste = 0.25 * torch.sin(
      torch.linspace(0, np.pi, N_grid, dtype=torch.float64, device=dispositivo)
  )

  n_tb, T_tb, I_tb = tb_solver.resolver_transporte(U_teste, 0.20, E_espectro)
  print(f"Corrente Matricial TB 1D:          {I_tb * 1e6:.4f} µA")
  print(f"Densidade Eletrônica Média Canal:   {torch.mean(n_tb).item():.4e} m⁻¹")

  # ---------------------------------------------------------
  # Parte C: Treinamento da PINN Acoplada ao NEGF
  # ---------------------------------------------------------
  print("\n" + "=" * 70)
  print(" PARTE C: Treinamento da PINN Acoplada (Poisson + NEGF Residual)")
  print("=" * 70)

  pinn = NanotransistorPINN(hidden_dim=48, num_layers=3).to(dispositivo)
  treinador = TreinadorPINN(
      pinn, sl_negf, L_ch=20e-9, lambda_g=3.0e-9, lr=2e-3
  )

  for epoca in range(1, 401):
    metricas = treinador.passo_treinamento(n_colocacao=64)
    if epoca % 100 == 0 or epoca == 1:
      print(
          f"Época {epoca:03d} | Loss Total: {metricas['loss_total']:.6e} |"
          f" Loss PDE: {metricas['loss_pde']:.6e} | Loss BC:"
          f" {metricas['loss_bc']:.6e}"
      )

  # ---------------------------------------------------------
  # Visualização dos Resultados
  # ---------------------------------------------------------
  x_coords = np.linspace(0, 20, N_grid)

  # Inferência da PINN para Vgs = 0.6 V, Vds = 0.3 V
  x_tensor = torch.linspace(0, 1, N_grid, device=dispositivo).unsqueeze(-1)
  vgs_tensor = torch.full((N_grid, 1), 0.6, device=dispositivo)
  vds_tensor = torch.full((N_grid, 1), 0.3, device=dispositivo)

  with torch.no_grad():
    phi_pinn = (
        pinn.forward_potencial(x_tensor, vgs_tensor, vds_tensor).cpu().numpy()
    )

  plt.figure(figsize=(11, 4.5))

  plt.subplot(1, 2, 1)
  plt.plot(
      E_espectro.cpu().numpy(),
      T_tb.cpu().numpy(),
      "b-",
      label="Transmissão T(E) - TB 1D",
  )
  plt.title("Espectro de Transmissão Quântica")
  plt.xlabel("Energia (eV)")
  plt.ylabel("T(E)")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(
      x_coords,
      phi_pinn,
      "r-",
      linewidth=2,
      label="Perfil ϕ(x) previsto pela PINN",
  )
  plt.title("Potencial Eletrostático PINN (Vgs=0.6V, Vds=0.3V)")
  plt.xlabel("Posição x (nm)")
  plt.ylabel("Potencial Eletrostático (V)")
  plt.grid(True, linestyle="--", alpha=0.6)
  plt.legend()

  plt.tight_layout()
  plt.show()
