"""Módulo: Plataforma Unificada de Nanoeletrônica (NEGF 1D, Poisson Reduzido & PINN)

Autor: Luiz Tiago Wilcke
Descrição: Framework completo com auto-consistência quântico-eletrostática rigorosa:
           1. Autoenergias exatas para contatos semi-infinitos 1D (Open Boundary
           Conditions).
           2. Solver 1D Tight-Binding NEGF matricial (G^R, G^<, A(x,E), T(E),
           I_DS).
           3. Eletrostática de Poisson Reduzida para DG-MOSFET com
           consistência dimensional (m⁻³).
           4. Loop Auto-Consistente (SCF) de referência com aceleração de
           convergência.
           5. PINN parametrizada phi(x, Vgs, Vds) com fechamento de ciclo
           quântico U(x) -> NEGF -> n(x).
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. CONSTANTES FÍSICAS E CONVERSÕES DIMENSIONAIS
# =============================================================================


class ConstantesFisicas:
  """Constantes fundamentais no SI e fatores de conversão."""

  Q_E: float = 1.602176634e-19  # C (Carga elementar)
  H_PLANCK: float = 6.62607015e-34  # J·s (Constante de Planck)
  H_BAR: float = 1.054571817e-34  # J·s (Planck reduzida)
  M_E: float = 9.1093837e-31  # kg (Massa do elétron livre)
  K_B_EV: float = 8.617333262e-5  # eV/K (Constante de Boltzmann)
  EPS_0: float = 8.8541878128e-12  # F/m (Permissividade do vácuo)
  EPS_SI: float = 11.7 * 8.8541878128e-12  # F/m (Permissividade do Silício)
  G0: float = 2.0 * (1.602176634e-19**2) / 6.62607015e-34  # ~77.48 µS (Quanta de condutância)


class GeometriaDispositivo:
  """Especificação geométrica do transistor para consistência dimensional estrita."""

  def __init__(
      self,
      L_canal: float = 15.0e-9,  # Comprimento do canal (m)
      W_largura: float = 5.0e-9,  # Largura do canal (m)
      T_corpo: float = 3.0e-9,  # Espessura do corpo semicondutor (m)
      lambda_gate: float = 2.5e-9,  # Comprimento característico de triagem da porta (m)
  ):
    self.L = L_canal
    self.W = W_largura
    self.T_body = T_corpo
    self.lambda_g = lambda_gate
    # Área da seção transversal efetiva: A_cross = W * T_body (m²)
    self.A_cross = self.W * self.T_body

  def densidade_1d_para_3d(self, n_1d: torch.Tensor) -> torch.Tensor:
    """Converte densidade linear [m⁻¹] para densidade volumétrica [m⁻³] sem fatores empíricos."""
    return n_1d / self.A_cross


# =============================================================================
# 2. AUTOENERGIAS DE CONTATO RIGOROSAS (OPEN BOUNDARY CONDITIONS)
# =============================================================================


class AutoEnergiaContato:
  """Cálculo analítico da autoenergia de contato para contatos semi-infinitos 1D

  (Condições de Contorno Abertas de Datta / Sanvito).
  """

  @staticmethod
  def calcular_sigma_lead(
      E: torch.Tensor, U_lead: float, t0: float
  ) -> torch.Tensor:
    """Calcula Sigma^R(E) para um contato 1D semi-infinito com potencial U_lead e hopping t0.

    Relação de dispersão no contato: E - U_lead - 2t0 = -2t0 * cos(k*a).
    """
    # Energia relativa normalizada
    theta = (E - U_lead - 2.0 * t0) / (2.0 * t0)

    # Inicialização da autoenergia complexa
    sigma = torch.zeros_like(E, dtype=torch.complex128)

    # 1. Estados propagantes dentro da banda de condução (|theta| <= 1)
    mask_band = torch.abs(theta) <= 1.0
    if mask_band.any():
      theta_b = theta[mask_band]
      # exp(i*k*a) = theta - i * sqrt(1 - theta²)
      # Sigma^R = -t0 * exp(i*k*a) = -t0*theta + i*t0*sqrt(1 - theta²)
      re_part = -t0 * theta_b
      im_part = -t0 * torch.sqrt(1.0 - theta_b**2)
      sigma[mask_band] = torch.complex(re_part, im_part)

    # 2. Estados evanescentes abaixo da banda (theta < -1)
    mask_below = theta < -1.0
    if mask_below.any():
      theta_bel = theta[mask_below]
      re_part = -t0 * (theta_bel + torch.sqrt(theta_bel**2 - 1.0))
      sigma[mask_below] = torch.complex(re_part, torch.zeros_like(re_part))

    # 3. Estados evanescentes acima da banda (theta > 1)
    mask_above = theta > 1.0
    if mask_above.any():
      theta_abv = theta[mask_above]
      re_part = -t0 * (theta_abv - torch.sqrt(theta_abv**2 - 1.0))
      sigma[mask_above] = torch.complex(re_part, torch.zeros_like(re_part))

    return sigma


# =============================================================================
# 3. SOLVER 1D TIGHT-BINDING NEGF MATRICIAL
# =============================================================================


class TightBindingNEGF1D:
  """Solver Quântico 1D baseado no Formalismo de Funções de Green Não-Equilíbrio."""

  def __init__(
      self,
      N_sites: int,
      dx: float,
      m_eff: float = 0.20,
      E_F: float = 0.0,
      T: float = 300.0,
      modo_contato: str = "rigoroso",  # 'rigoroso' (Lead 1D) ou 'wbl' (Wide-Band)
      gamma_wbl: float = 0.15,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.m_eff = m_eff
    self.E_F = E_F
    self.T = T
    self.modo_contato = modo_contato
    self.gamma_wbl = gamma_wbl
    self.device = device
    self.kBT = max(ConstantesFisicas.K_B_EV * self.T, 1e-7)

    # Parâmetro de hopping: t0 = hbar² / (2 * m* * dx²) em eV
    t0_joules = (ConstantesFisicas.H_BAR**2) / (
        2.0 * (self.m_eff * ConstantesFisicas.M_E) * (self.dx**2)
    )
    self.t0 = float(t0_joules / ConstantesFisicas.Q_E)

    # Hamiltoniano cinético tridiagonal H_kin
    diag_kin = 2.0 * self.t0 * torch.ones(self.N, dtype=torch.float64)
    off_kin = -self.t0 * torch.ones(self.N - 1, dtype=torch.float64)
    self.H_kin = (
        torch.diag(diag_kin) + torch.diag(off_kin, 1) + torch.diag(off_kin, -1)
    ).to(self.device)

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
    """Calcula densidade linear n_1D(x) [m⁻¹], Transmissão T(E) e Corrente I_DS [A].

    G^R(E) = [E*I - H - Sigma_S(E) - Sigma_D(E)]⁻¹
    G^<(E) = G^R * (Gamma_S*f_S + Gamma_D*f_D) * G^A
    """
    H = self.H_kin + torch.diag(U_potencial.to(torch.float64))
    H_c = H.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    U_S = float(U_potencial[0].item())
    U_D = float(U_potencial[-1].item())

    T_lista = []
    n_1d = torch.zeros(self.N, dtype=torch.float64, device=self.device)

    for E in E_grid:
      E_c = E.to(torch.complex128)

      # 1. Construção das Autoenergias de Contato
      Sigma_S_mat = torch.zeros(
          (self.N, self.N), dtype=torch.complex128, device=self.device
      )
      Sigma_D_mat = torch.zeros(
          (self.N, self.N), dtype=torch.complex128, device=self.device
      )

      if self.modo_contato == "rigoroso":
        sig_S = AutoEnergiaContato.calcular_sigma_lead(E, U_S, self.t0)
        sig_D = AutoEnergiaContato.calcular_sigma_lead(E, U_D, self.t0)
        Sigma_S_mat[0, 0] = sig_S
        Sigma_D_mat[-1, -1] = sig_D
      else:
        Sigma_S_mat[0, 0] = -1j * (self.gamma_wbl / 2.0)
        Sigma_D_mat[-1, -1] = -1j * (self.gamma_wbl / 2.0)

      Gamma_S = 1j * (Sigma_S_mat - Sigma_S_mat.conj().T)
      Gamma_D = 1j * (Sigma_D_mat - Sigma_D_mat.conj().T)

      f_S = self.fermi_dirac(E, mu_S)
      f_D = self.fermi_dirac(E, mu_D)

      # 2. Inversão da Função de Green Retardada
      A_mat = (E_c + 1e-7j) * self.I_mat - H_c - Sigma_S_mat - Sigma_D_mat
      G_R = torch.linalg.inv(A_mat)
      G_A = G_R.conj().T

      # 3. Transmissão Quântica T(E) = Tr[Gamma_S * G_R * Gamma_D * G_A]
      T_E = torch.trace(Gamma_S @ G_R @ Gamma_D @ G_A).real
      T_lista.append(torch.clamp(T_E, min=0.0))

      # 4. Função de Correlação Menor G^<(E)
      Sigma_in = Gamma_S * f_S + Gamma_D * f_D
      G_lesser = G_R @ Sigma_in @ G_A

      # Densidade de carga por intervalo de energia: n_1D(x, E) = diag(G^<) / (2*pi*dx)
      n_1d += torch.diag(G_lesser).imag / (2.0 * np.pi * self.dx)

    dE = (E_grid[1] - E_grid[0]).item()
    n_1d = n_1d * dE
    T_espectro = torch.stack(T_lista)

    # 5. Corrente Terminal de Landauer-Büttiker
    f_S_all = self.fermi_dirac(E_grid, mu_S)
    f_D_all = self.fermi_dirac(E_grid, mu_D)
    integrando_I = T_espectro * (f_S_all - f_D_all)
    I_ds = float(
        ConstantesFisicas.G0 * torch.trapezoid(integrando_I, E_grid).item()
    )

    return n_1d, T_espectro, I_ds


# =============================================================================
# 4. EQUAÇÃO DE POISSON REDUZIDA COM ACOPLAMENTO DE PORTA
# =============================================================================


class PoissonReduzido1D:
  """Modelo Eletrostático Reduzido de Porta para DG-MOSFET (Thin-Body Approximation):

  d²phi/dx² - (phi(x) - Vgs)/lambda_g² = -q * (N_dop(x) - n_3D(x)) / eps_semi
  """

  def __init__(
      self,
      N_sites: int,
      dx: float,
      lambda_g: float,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.lambda_sq = lambda_g**2
    self.device = device

    # Operador diferencial linear: d²/dx² - 1/lambda_g²
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
      n_3d: torch.Tensor,
      N_dop_3d: torch.Tensor,
      Vgs: float,
      Vs: float,
      Vd: float,
  ) -> torch.Tensor:
    """Resolve a distribuição contínua de potencial phi(x) em Volts."""
    rho = ConstantesFisicas.Q_E * (N_dop_3d - n_3d)
    RHS = -(rho / ConstantesFisicas.EPS_SI) - (Vgs / self.lambda_sq)

    A = self.M_poisson.clone()
    b = RHS.clone()

    # Condições de Contorno de Dirichlet nos contatos ôhmicos
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = Vs

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = Vd

    return torch.linalg.solve(A, b)


# =============================================================================
# 5. LOOP AUTO-CONSISTENTE POISSON-NEGF (SCF REFERENCE BENCHMARK)
# =============================================================================


class AutoConsistenteSCF:
  """Loop de Ponto Fixo (Self-Consistent Field) com amortecimento adaptativo de Picard."""

  def __init__(
      self,
      negf_solver: TightBindingNEGF1D,
      poisson_solver: PoissonReduzido1D,
      geometria: GeometriaDispositivo,
      max_iter: int = 60,
      tol: float = 1e-4,
      alpha_mix: float = 0.12,
  ):
    self.negf = negf_solver
    self.poisson = poisson_solver
    self.geom = geometria
    self.max_iter = max_iter
    self.tol = tol
    self.alpha = alpha_mix

  def resolver(
      self,
      Vgs: float,
      Vds: float,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
  ) -> Dict[str, torch.Tensor]:
    """Executa a auto-consistência: phi -> U(x) -> NEGF -> n_1D -> n_3D -> Poisson -> phi_novo."""
    # Chute inicial linear de potencial
    phi = torch.linspace(
        0.0, Vds, self.negf.N, dtype=torch.float64, device=self.negf.device
    )
    n_3d = torch.zeros_like(N_dop_3d)

    for it in range(self.max_iter):
      # Energia Potencial Eletrostática: U(x) = -q * phi(x) em eV
      U_pot = -phi

      # 1. Passo Quântico NEGF
      n_1d, T_E, I_ds = self.negf.resolver_transporte(U_pot, Vds, E_grid)
      n_3d_novo = self.geom.densidade_1d_para_3d(n_1d)

      # 2. Passo Eletrostático de Poisson
      phi_novo = self.poisson.resolver(n_3d_novo, N_dop_3d, Vgs, 0.0, Vds)

      # 3. Verificação de Convergência e Mistura (Picard Damping)
      res_max = torch.max(torch.abs(phi_novo - phi)).item()
      phi = (1.0 - self.alpha) * phi + self.alpha * phi_novo
      n_3d = (1.0 - self.alpha) * n_3d + self.alpha * n_3d_novo

      if res_max < self.tol:
        break

    return {
        "phi": phi,
        "U_x": -phi,
        "n_3d": n_3d,
        "n_1d": n_1d,
        "T_E": T_E,
        "I_ds": I_ds,
        "iter": it + 1,
        "convergido": res_max < self.tol,
    }


# =============================================================================
# 6. PINN PARAMÉTRICA COM ACOPLAMENTO QUÂNTICO COMPLETO
# =============================================================================


class TransistorPINN(nn.Module):
  """Surrogate Neural Paramétrico que mapeia (x, Vgs, Vds) -> phi(x, Vgs, Vds)."""

  def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
    super().__init__()
    camadas = [nn.Linear(3, hidden_dim), nn.Tanh()]
    for _ in range(num_layers - 1):
      camadas.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
    camadas.append(nn.Linear(hidden_dim, 1))
    self.rede = nn.Sequential(*camadas)

  def forward(
      self, x_norm: torch.Tensor, Vgs: torch.Tensor, Vds: torch.Tensor
  ) -> torch.Tensor:
    """x_norm: x/L normalizado em [0, 1]."""
    entradas = torch.cat([x_norm, Vgs, Vds], dim=-1)
    return self.rede(entradas)


class TreinadorPINNAutoConsistente:
  """Treinador com fechamento de ciclo: phi_pred -> U(x) -> NEGF -> n_3D(x) -> PDE Loss."""

  def __init__(
      self,
      pinn: TransistorPINN,
      negf_solver: TightBindingNEGF1D,
      geometria: GeometriaDispositivo,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
      lr: float = 1e-3,
  ):
    self.pinn = pinn
    self.negf = negf_solver
    self.geom = geometria
    self.N_dop = N_dop_3d
    self.E_grid = E_grid
    self.opt = optim.Adam(self.pinn.parameters(), lr=lr)

  def passo_treinamento_fechado(
      self, Vgs_val: float, Vds_val: float
  ) -> Dict[str, float]:
    """Passo de otimização fechando a dependência do resíduo de Poisson na densidade NEGF real."""
    self.opt.zero_grad()

    N_pts = self.negf.N
    x_real = torch.linspace(
        0, self.geom.L, N_pts, dtype=torch.float32, requires_grad=True
    ).unsqueeze(-1)
    x_norm = x_real / self.geom.L
    vgs_t = torch.full((N_pts, 1), Vgs_val, dtype=torch.float32)
    vds_t = torch.full((N_pts, 1), Vds_val, dtype=torch.float32)

    # 1. Inferência da PINN para o Perfil de Potencial
    phi_pred = self.pinn(x_norm, vgs_t, vds_t)

    # 2. Gradientes Eletrostáticos Exatos via Autograd
    dphi_dx = torch.autograd.grad(
        phi_pred,
        x_real,
        grad_outputs=torch.ones_like(phi_pred),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2phi_dx2 = torch.autograd.grad(
        dphi_dx,
        x_real,
        grad_outputs=torch.ones_like(dphi_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    # 3. Fechamento de Ciclo Quântico: U(x) = -q * phi_pred -> NEGF 1D
    # Converte para float64 para estabilidade do solver matricial
    U_pot_quântico = -phi_pred.detach().squeeze(-1).to(torch.float64)
    n_1d, _, _ = self.negf.resolver_transporte(
        U_pot_quântico, Vds_val, self.E_grid
    )
    n_3d_quântico = (
        self.geom.densidade_1d_para_3d(n_1d).unsqueeze(-1).to(torch.float32)
    )

    # 4. Resíduo de Poisson Reduzido com Carga Quântica Exata
    rho = ConstantesFisicas.Q_E * (self.N_dop.unsqueeze(-1) - n_3d_quântico)
    res_poisson = (
        d2phi_dx2
        - ((phi_pred - Vgs_val) / (self.geom.lambda_g**2))
        + (rho / ConstantesFisicas.EPS_SI)
    )
    loss_pde = torch.mean(res_poisson**2) * 1e-18  # Escala para regularização

    # 5. Perdas de Contorno nos Eletrodos (Dirichlet)
    loss_bc_source = (phi_pred[0, 0] - 0.0) ** 2
    loss_bc_drain = (phi_pred[-1, 0] - Vds_val) ** 2
    loss_bc = loss_bc_source + loss_bc_drain

    # 6. Otimização Conjunta
    loss_total = loss_pde + 20.0 * loss_bc
    loss_total.backward()
    self.opt.step()

    return {
        "loss_total": loss_total.item(),
        "loss_pde": loss_pde.item(),
        "loss_bc": loss_bc.item(),
    }


# =============================================================================
# 7. EXECUÇÃO, BENCHMARK E VISUALIZAÇÃO COMPLETA
# =============================================================================

if __name__ == "__main__":
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("=" * 75)
  print(
      " PLATAFORMA QUÂNTICO-ELETROSTÁTICA COMPLETA: NEGF 1D + POISSON + PINN"
  )
  print("=" * 75)

  # Geometria e Discretização
  geom = GeometriaDispositivo(
      L_canal=15.0e-9, W_largura=5.0e-9, T_corpo=3.0e-9, lambda_gate=2.2e-9
  )
  N_grid = 45
  dx = geom.L / (N_grid - 1)

  # Perfil Físico de Dopagem Tridimensional N+/i/N+ (m⁻³)
  N_dop_3d = torch.zeros(N_grid, dtype=torch.float64, device=dispositivo)
  idx_contato = int(0.20 * N_grid)
  N_dop_3d[:idx_contato] = 1e26  # Fonte altamente dopada (1e20 cm⁻³)
  N_dop_3d[-idx_contato:] = 1e26  # Dreno altamente dopado
  N_dop_3d[idx_contato:-idx_contato] = 1e21  # Canal quase intrínseco (1e15 cm⁻³)

  # Solvers Físicos
  negf = TightBindingNEGF1D(
      N_sites=N_grid,
      dx=dx,
      m_eff=0.20,
      E_F=0.0,
      T=300.0,
      modo_contato="rigoroso",
      device=dispositivo,
  )
  poisson = PoissonReduzido1D(
      N_sites=N_grid, dx=dx, lambda_g=geom.lambda_g, device=dispositivo
  )
  scf = AutoConsistenteSCF(
      negf, poisson, geom, max_iter=70, tol=1e-4, alpha_mix=0.15
  )

  # Grid de Integração de Energia
  E_grid = torch.linspace(
      -0.4, 1.4, 350, dtype=torch.float64, device=dispositivo
  )

  # --- PARTE A: Solução de Referência via SCF ---
  print("\n>>> [1/3] Executando Solver Auto-Consistente (SCF) de Referência...")
  Vgs_alvo, Vds_alvo = 0.50, 0.30
  res_scf = scf.resolver(Vgs_alvo, Vds_alvo, N_dop_3d, E_grid)

  print(
      f"SCF Convergido: {res_scf['convergido']} em {res_scf['iter']} iterações"
  )
  print(f"Corrente Landauer SCF (Ids): {res_scf['I_ds'] * 1e6:.4f} µA")
  print(
      f"Densidade 3D Pico no Canal:  {torch.max(res_scf['n_3d']).item():.3e} m⁻³"
  )

  # --- PARTE B: Treinamento da PINN Fechada com NEGF ---
  print("\n>>> [2/3] Treinando PINN Paramétrica com Acoplamento Fechado...")
  pinn_model = TransistorPINN(hidden_dim=48, num_layers=4).to(dispositivo)
  treinador = TreinadorPINNAutoConsistente(
      pinn_model,
      negf,
      geom,
      N_dop_3d.to(torch.float32),
      E_grid,
      lr=1.5e-3,
  )

  for ep in range(1, 201):
    metricas = treinador.passo_treinamento_fechado(Vgs_alvo, Vds_alvo)
    if ep % 50 == 0 or ep == 1:
      print(
          f"Época {ep:03d} | Loss Total: {metricas['loss_total']:.5e} | PDE:"
          f" {metricas['loss_pde']:.5e} | BC: {metricas['loss_bc']:.5e}"
      )

  # Inferência da PINN treinada
  x_eval = (
      torch.linspace(0, 1, N_grid, device=dispositivo)
      .unsqueeze(-1)
      .to(torch.float32)
  )
  vgs_eval = torch.full((N_grid, 1), Vgs_alvo, device=dispositivo, dtype=torch.float32)
  vds_eval = torch.full((N_grid, 1), Vds_alvo, device=dispositivo, dtype=torch.float32)

  with torch.no_grad():
    phi_pinn = pinn_model(x_eval, vgs_eval, vds_eval).cpu().numpy().squeeze()

  # --- PARTE C: Visualização Comparativa e Espectro Quântico ---
  print("\n>>> [3/3] Gerando visualizações físicas...")
  x_nm = np.linspace(0, geom.L * 1e9, N_grid)

  fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))

  # 1. Comparação de Potencial Eletrostático
  axs[0].plot(
      x_nm,
      res_scf["phi"].cpu().numpy(),
      "b-",
      linewidth=2.2,
      label="SCF Exato",
  )
  axs[0].plot(
      x_nm,
      phi_pinn,
      "r--",
      linewidth=2.0,
      label="PINN Fechada",
  )
  axs[0].set_title(f"Potencial phi(x) (Vgs={Vgs_alvo}V, Vds={Vds_alvo}V)")
  axs[0].set_xlabel("Posição x (nm)")
  axs[0].set_ylabel("Potencial (V)")
  axs[0].grid(True, linestyle="--", alpha=0.6)
  axs[0].legend()

  # 2. Densidade Volumétrica de Portadores
  axs[1].semilogy(
      x_nm,
      res_scf["n_3d"].cpu().numpy() + 1e18,
      "g-",
      linewidth=2,
      label="n_3D(x) [m⁻³]",
  )
  axs[1].semilogy(
      x_nm,
      N_dop_3d.cpu().numpy(),
      "k:",
      alpha=0.7,
      label="Dopagem N_dop [m⁻³]",
  )
  axs[1].set_title("Perfil de Densidade de Carga")
  axs[1].set_xlabel("Posição x (nm)")
  axs[1].set_ylabel("Concentração (m⁻³)")
  axs[1].grid(True, linestyle="--", alpha=0.6)
  axs[1].legend()

  # 3. Transmissão Quântica T(E)
  axs[2].plot(
      E_grid.cpu().numpy(),
      res_scf["T_E"].cpu().numpy(),
      "m-",
      linewidth=2,
      label="Transmissão T(E)",
  )
  axs[2].axvline(
      negf.E_F,
      color="blue",
      linestyle="--",
      label="mu_Source (0.0 eV)",
  )
  axs[2].axvline(
      negf.E_F - Vds_alvo,
      color="red",
      linestyle="--",
      label=f"mu_Drain (-{Vds_alvo} eV)",
  )
  axs[2].set_title("Espectro de Transmissão Quântica T(E)")
  axs[2].set_xlabel("Energia E (eV)")
  axs[2].set_ylabel("T(E)")
  axs[2].grid(True, linestyle="--", alpha=0.6)
  axs[2].legend()

  plt.tight_layout()
  plt.show()
