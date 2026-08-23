"""Framework Unificado de Nanoeletrônica Computacional e Scientific ML

Módulo: Multi-Scale Quantum & Semiclassical Transport with Bayesian PINN UQ
Autor: Luiz Tiago Wilcke
Descrição: Plataforma completa integrando:
           1. Eletrostática: Poisson 1D reduzido (acoplamento de porta DG-MOSFET / Nanowire).
           2. Transporte Quântico: 1D Tight-Binding NEGF com autoenergias analíticas de contatos
              semi-infinitos (OBC), cálculo vetorizado de G^R(E), G^<(E), A(x,E), T(E) e I_DS.
           3. Transporte Semiclássico: Drift-Diffusion 1D acoplado para comparação multiescala.
           4. Loop Auto-Consistente (SCF): Solver de Ponto Fixo de referência com amortecimento de Picard.
           5. Physics-Informed Neural Network Bayesiana (MCDropoutPINN): Surrogate paramétrico phi(x, Vgs, Vds).
           6. Propagação Quântica de Incerteza (UQ End-to-End):
              phi^(k)(x) -> U^(k)(x) -> G^R,(k) -> T^(k)(E) -> I_D^(k),
              gerando bandas de confiança bayesianas completas para potenciais, densidades e curvas Id-Vg/Id-Vd.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. CONSTANTES FÍSICAS UNIVERSAIS E GEOMETRIA DO DISPOSITIVO
# =============================================================================


class ConstantesFisicas:
  """Constantes fundamentais no Sistema Internacional (SI) e fatores quânticos."""

  Q_E: float = 1.602176634e-19  # C (Carga elementar)
  H_PLANCK: float = 6.62607015e-34  # J·s (Constante de Planck)
  H_BAR: float = 1.054571817e-34  # J·s (Planck reduzida)
  M_E: float = 9.1093837e-31  # kg (Massa do elétron livre)
  K_B: float = 1.380649e-23  # J/K (Constante de Boltzmann no SI)
  K_B_EV: float = 8.617333262e-5  # eV/K (Boltzmann em eV)
  EPS_0: float = 8.8541878128e-12  # F/m (Permissividade do vácuo)
  EPS_SI: float = (
      11.7 * 8.8541878128e-12
  )  # F/m (Permissividade do Silício: eps_r = 11.7)
  G0: float = (
      2.0 * (1.602176634e-19**2) / 6.62607015e-34
  )  # 2q²/h ≈ 7.74809e-5 S (A/eV)


class GeometriaDispositivo:
  """Especificação geométrica com consistência dimensional estrita."""

  def __init__(
      self,
      L_canal: float = 15.0e-9,  # Comprimento do canal (m) -> 15 nm
      W_largura: float = 5.0e-9,  # Largura do canal (m) -> 5 nm
      T_corpo: float = 3.0e-9,  # Espessura do corpo (m) -> 3 nm
      lambda_gate: float = 2.2e-9,  # Comprimento de triagem da porta (m)
  ):
    self.L = L_canal
    self.W = W_largura
    self.T_body = T_corpo
    self.lambda_g = lambda_gate
    # Área da seção transversal perpendicular ao transporte: A = W * T_body (m²)
    self.A_cross = self.W * self.T_body

  def densidade_1d_para_3d(self, n_1d: torch.Tensor) -> torch.Tensor:
    """Converte densidade linear [m⁻¹] para volumétrica [m⁻³]: n_3D = n_1D / A_cross."""
    return n_1d / self.A_cross

  def densidade_3d_para_1d(self, n_3d: torch.Tensor) -> torch.Tensor:
    """Converte densidade volumétrica [m⁻³] para linear [m⁻¹]: n_1D = n_3D * A_cross."""
    return n_3d * self.A_cross


# =============================================================================
# 2. AUTOENERGIAS ANALÍTICAS DE CONTATO 1D (OPEN BOUNDARY CONDITIONS)
# =============================================================================


class AutoEnergiaLead1D:
  """Autoenergia analítica exata para contatos 1D semi-infinitos (Open Boundary Conditions).

  Relação de dispersão do contato: E - U_lead - 2t0 = -2t0 * cos(k*a).
  """

  @staticmethod
  def calcular_sigma(
      E_grid: torch.Tensor, U_lead: float, t0: float
  ) -> torch.Tensor:
    """Calcula Sigma^R(E) para todo o grid espectral de energias."""
    theta = (E_grid - U_lead - 2.0 * t0) / (2.0 * t0)
    sigma = torch.zeros_like(E_grid, dtype=torch.complex128)

    # 1. Estados propagantes dentro da banda de condução (|theta| <= 1)
    mask_band = torch.abs(theta) <= 1.0
    if mask_band.any():
      th_b = theta[mask_band]
      re_part = t0 * th_b
      im_part = -t0 * torch.sqrt(1.0 - th_b**2)
      sigma[mask_band] = torch.complex(re_part, im_part)

    # 2. Estados evanescentes abaixo da banda (theta < -1)
    mask_below = theta < -1.0
    if mask_below.any():
      th_bel = theta[mask_below]
      re_part = t0 * (th_bel + torch.sqrt(th_bel**2 - 1.0))
      sigma[mask_below] = torch.complex(re_part, torch.zeros_like(re_part))

    # 3. Estados evanescentes acima da banda (theta > 1)
    mask_above = theta > 1.0
    if mask_above.any():
      th_abv = theta[mask_above]
      re_part = t0 * (th_abv - torch.sqrt(th_abv**2 - 1.0))
      sigma[mask_above] = torch.complex(re_part, torch.zeros_like(re_part))

    return sigma


# =============================================================================
# 3. SOLVER 1D TIGHT-BINDING NEGF MATRICIAL (LINEAR SOLVE VECTORIZADO)
# =============================================================================


class TightBindingNEGF1D:
  """Solver Quântico 1D baseado em Funções de Green Não-Equilíbrio (NEGF).

  Calcula G^R(E), G^<(E), A(x,E), T(E) e I_DS sem inversões explícitas inv(A).
  """

  def __init__(
      self,
      N_sites: int,
      dx: float,
      m_eff: float = 0.20,
      E_F: float = 0.0,
      T: float = 300.0,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.m_eff = m_eff
    self.E_F = E_F
    self.T = T
    self.kBT = max(ConstantesFisicas.K_B_EV * self.T, 1e-7)
    self.device = device

    # Parâmetro de hopping cinético: t0 = hbar² / (2 * m* * dx²) em eV
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
    """Ocupação de Fermi-Dirac protegida numericamente."""
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def resolver_transporte(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Calcula n_1D(x) [m⁻¹], T(E), I_DS [A] e LDOS A(x,E) [eV⁻¹·m⁻¹]."""
    n_energies = E_grid.shape[0]
    H_total = self.H_kin + torch.diag(U_potencial.to(torch.float64))
    H_c = H_total.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    U_S = float(U_potencial[0].item())
    U_D = float(U_potencial[-1].item())

    # 1. Autoenergias analíticas dos contatos
    sigma_S_vec = AutoEnergiaLead1D.calcular_sigma(E_grid, U_S, self.t0)
    sigma_D_vec = AutoEnergiaLead1D.calcular_sigma(E_grid, U_D, self.t0)

    # 2. Montagem do sistema linear em lote: A_batch(E) @ G^R(E) = I
    A_batch = (
        (E_grid.to(torch.complex128) + 1e-7j).view(-1, 1, 1)
        * self.I_mat.unsqueeze(0)
    ) - H_c.unsqueeze(0)
    A_batch = A_batch.clone()
    A_batch[:, 0, 0] -= sigma_S_vec
    A_batch[:, -1, -1] -= sigma_D_vec

    I_expanded = self.I_mat.unsqueeze(0).expand(n_energies, self.N, self.N)
    G_R_batch = torch.linalg.solve(A_batch, I_expanded)

    gamma_S_val = -2.0 * sigma_S_vec.imag
    gamma_D_val = -2.0 * sigma_D_vec.imag

    # 3. Transmissão Quântica T(E) = Gamma_S * Gamma_D * |G^R[0, N-1]|²
    G_R_0N = G_R_batch[:, 0, -1]
    T_E = torch.clamp(
        gamma_S_val * gamma_D_val * (G_R_0N.real**2 + G_R_0N.imag**2), min=0.0
    )

    f_S = self.fermi_dirac(E_grid, mu_S)
    f_D = self.fermi_dirac(E_grid, mu_D)

    # 4. Função de Green Menor: G^<(E) = G^R @ Sigma^in @ G^A
    G_R_i0_sq = (
        G_R_batch[:, :, 0].real ** 2 + G_R_batch[:, :, 0].imag ** 2
    )  # [n_E, N]
    G_R_iN_sq = (
        G_R_batch[:, :, -1].real ** 2 + G_R_batch[:, :, -1].imag ** 2
    )  # [n_E, N]

    G_lesser_diag = G_R_i0_sq * (gamma_S_val * f_S).unsqueeze(
        -1
    ) + G_R_iN_sq * (gamma_D_val * f_D).unsqueeze(-1)

    # 5. Densidade Local de Estados (LDOS): A(x, E) = -(1/pi) * Im[diag(G^R)] / dx
    ldos = -(1.0 / np.pi) * (G_R_batch.diagonal(dim1=-2, dim2=-1).imag) / self.dx

    # 6. Densidade Linear n_1D(x) integrada em energia (com spin gs=2, vale gv=1)
    dE = (E_grid[1] - E_grid[0]).item()
    integrando_n = (2.0 / (2.0 * np.pi * self.dx)) * G_lesser_diag
    n_1d = torch.sum(integrando_n, dim=0) * dE

    # 7. Corrente Landauer-Büttiker Terminal (Amperes)
    integrando_I = T_E * (f_S - f_D)
    I_ds = float(
        ConstantesFisicas.G0 * torch.trapezoid(integrando_I, E_grid).item()
    )

    return n_1d, T_E, I_ds, ldos


# =============================================================================
# 4. SOLVER SEMICLÁSSICO: DRIFT-DIFFUSION 1D (SCHARFETTER-GUMMEL)
# =============================================================================


class DriftDiffusion1D:
  """Solver Semiclássico 1D baseado na equação da continuidade e formulação de Scharfetter-Gummel."""

  def __init__(
      self,
      N_sites: int,
      dx: float,
      geom: GeometriaDispositivo,
      mu_n: float = 0.05,  # Mobilidade dos elétrons (m²/(V·s))
      T: float = 300.0,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.geom = geom
    self.mu_n = mu_n
    self.T = T
    self.Vt = (ConstantesFisicas.K_B * self.T) / ConstantesFisicas.Q_E  # ~0.0259 V
    self.Dn = self.mu_n * self.Vt  # Relação de Einstein (m²/s)
    self.device = device

  def bernoulli(self, x: torch.Tensor) -> torch.Tensor:
    """Função de Bernoulli B(x) = x / (exp(x) - 1) com estabilidade para x -> 0."""
    x_clamped = torch.clamp(x, -40.0, 40.0)
    pequeno = torch.abs(x_clamped) < 1e-4
    res = torch.zeros_like(x_clamped)
    res[pequeno] = 1.0 - 0.5 * x_clamped[pequeno]
    res[~pequeno] = x_clamped[~pequeno] / (torch.exp(x_clamped[~pequeno]) - 1.0)
    return res

  def resolver_transporte(
      self,
      phi: torch.Tensor,
      N_dop_3d: torch.Tensor,
  ) -> Tuple[torch.Tensor, float]:
    """Resolve a densidade estacionária n_3D(x) e a corrente de dreno semiclássica I_DD [A]."""
    # Diferença de potencial adimensionalizada entre sítios adjacentes
    delta_phi = (phi[1:] - phi[:-1]) / self.Vt
    B_pos = self.bernoulli(delta_phi)
    B_neg = self.bernoulli(-delta_phi)

    # Montagem da matriz tridiagonal de continuidade estacionária: dJn/dx = 0
    A = torch.zeros((self.N, self.N), dtype=torch.float64, device=self.device)
    b = torch.zeros((self.N,), dtype=torch.float64, device=self.device)

    for i in range(1, self.N - 1):
      A[i, i - 1] = B_neg[i - 1]
      A[i, i] = -(B_pos[i - 1] + B_neg[i])
      A[i, i + 1] = B_pos[i]

    # Condições de Contorno de Equilíbrio nos Contatos Ôhmicos: n = N_dop
    A[0, 0] = 1.0
    b[0] = N_dop_3d[0]
    A[-1, -1] = 1.0
    b[-1] = N_dop_3d[-1]

    n_3d = torch.linalg.solve(A, b)

    # Densidade de corrente média Jn = q * Dn/dx * [n_{i+1}*B(-dphi) - n_i*B(dphi)]
    Jn_local = (
        ConstantesFisicas.Q_E
        * (self.Dn / self.dx)
        * (n_3d[1:] * B_neg - n_3d[:-1] * B_pos)
    )
    Jn_medio = torch.mean(Jn_local).item()
    I_dd = float(Jn_medio * self.geom.A_cross)  # Corrente em Amperes (A)

    return n_3d, I_dd


# =============================================================================
# 5. POISSON REDUZIDO 1D E LOOP AUTO-CONSISTENTE (SCF REFERENCE BENCHMARK)
# =============================================================================


class PoissonReduzido1D:
  """Equação de Poisson 1D com modelo de triagem de porta (DG-MOSFET / Nanowire):

  d²phi/dx² - (phi(x) - Vgs)/lambda_g² = -q * (N_dop(x) - n_3D(x)) / eps_si
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
    """Resolve a distribuição contínua de potencial phi(x) em Volts via Dirichlet BCs."""
    rho = ConstantesFisicas.Q_E * (N_dop_3d - n_3d)
    RHS = -(rho / ConstantesFisicas.EPS_SI) - (Vgs / self.lambda_sq)

    A = self.M_poisson.clone()
    b = RHS.clone()

    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = Vs

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = Vd

    return torch.linalg.solve(A, b)


class AutoConsistenteSCF:
  """Solver de Ponto Fixo Auto-Consistente (Self-Consistent Field) com amortecimento Picard."""

  def __init__(
      self,
      negf_solver: TightBindingNEGF1D,
      poisson_solver: PoissonReduzido1D,
      geometria: GeometriaDispositivo,
      max_iter: int = 80,
      tol: float = 1e-4,
      alpha_mix: float = 0.12,
  ):
    self.negf = negf_solver
    self.poisson = poisson_solver
    self.geom = geometria
    self.max_iter = max_iter
    self.tol = tol
    self.alpha = alpha_mix

  def executar(
      self,
      Vgs: float,
      Vds: float,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
  ) -> Dict[str, torch.Tensor]:
    """Executa o ciclo SCF: phi -> U(x) -> NEGF -> n_3D -> Poisson -> phi_novo."""
    phi = torch.linspace(
        0.0, Vds, self.negf.N, dtype=torch.float64, device=self.negf.device
    )
    n_3d = torch.zeros_like(N_dop_3d)

    for it in range(self.max_iter):
      U_pot = -phi  # U(x) = -q * phi(x) em eV

      # 1. Transporte Quântico NEGF
      n_1d, T_E, I_ds, ldos = self.negf.resolver_transporte(U_pot, Vds, E_grid)
      n_3d_novo = self.geom.densidade_1d_para_3d(n_1d)

      # 2. Eletrostática de Poisson
      phi_novo = self.poisson.resolver(n_3d_novo, N_dop_3d, Vgs, 0.0, Vds)

      # 3. Convergência e Mistura
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
        "ldos": ldos,
        "iter": it + 1,
        "convergido": res_max < self.tol,
    }


# =============================================================================
# 6. BAYESIAN PINN (MC DROPOUT COM PRESERVAÇÃO DE ESTADO)
# =============================================================================


class MCDropoutPINN(nn.Module):
  """Physics-Informed Neural Network Bayesiana aproximada via Monte Carlo Dropout.

  Mapeia (x_norm, Vgs, Vds) -> phi(x, Vgs, Vds) permitindo amostragem variacional estocástica.
  """

  def __init__(
      self,
      in_dim: int = 3,
      hidden_dim: int = 64,
      num_layers: int = 4,
      p_dropout: float = 0.10,
  ):
    super().__init__()
    self.p_dropout = p_dropout
    camadas = [
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=p_dropout),
    ]
    for _ in range(num_layers - 1):
      camadas.extend([
          nn.Linear(hidden_dim, hidden_dim),
          nn.Tanh(),
          nn.Dropout(p=p_dropout),
      ])
    camadas.append(nn.Linear(hidden_dim, 1))
    self.rede = nn.Sequential(*camadas)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Forward pass padrão."""
    return self.rede(x)

  def amostrar_potenciais(
      self,
      x_norm: torch.Tensor,
      vgs_tensor: torch.Tensor,
      vds_tensor: torch.Tensor,
      n_amostras: int = 50,
  ) -> torch.Tensor:
    """Amostra N realizações estocásticas phi^(k)(x) preservando o estado do grafo."""
    estado_original = self.training
    self.train()  # Ativa o dropout para amostragem estocástica

    inputs = torch.cat([x_norm, vgs_tensor, vds_tensor], dim=-1)

    with torch.no_grad():
      amostras = torch.stack(
          [self.forward(inputs).squeeze(-1) for _ in range(n_amostras)], dim=0
      )

    self.train(estado_original)  # Restaura o estado original de treinamento
    return amostras  # [n_amostras, N_pontos]


class TreinadorPINNAutoConsistente:
  """Treinamento da PINN acoplado ao ciclo quântico real U(x) -> NEGF -> n_3D -> PDE Loss."""

  def __init__(
      self,
      pinn: MCDropoutPINN,
      negf: TightBindingNEGF1D,
      geom: GeometriaDispositivo,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
      lr: float = 2e-3,
      peso_decay: float = 1e-5,
  ):
    self.pinn = pinn
    self.negf = negf
    self.geom = geom
    self.N_dop = N_dop_3d
    self.E_grid = E_grid
    self.opt = optim.Adam(self.pinn.parameters(), lr=lr, weight_decay=peso_decay)

  def passo_treinamento(self, Vgs_val: float, Vds_val: float) -> Dict[str, float]:
    """Executa um passo de otimização estocástica com resíduo diferencial exato."""
    self.opt.zero_grad()
    self.pinn.train()

    N_pts = self.negf.N
    x_real = torch.linspace(
        0, self.geom.L, N_pts, dtype=torch.float32, requires_grad=True
    ).unsqueeze(-1)
    x_norm = x_real / self.geom.L
    vgs_t = torch.full((N_pts, 1), Vgs_val, dtype=torch.float32)
    vds_t = torch.full((N_pts, 1), Vds_val, dtype=torch.float32)

    inputs = torch.cat([x_norm, vgs_t, vds_t], dim=-1)
    phi_pred = self.pinn(inputs)

    # Gradientes de Poisson via Autograd
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

    # Carga quântica acoplada via NEGF 1D
    U_quântico = -phi_pred.detach().squeeze(-1).to(torch.float64)
    n_1d, _, _, _ = self.negf.resolver_transporte(
        U_quântico, Vds_val, self.E_grid
    )
    n_3d = self.geom.densidade_1d_para_3d(n_1d).unsqueeze(-1).to(torch.float32)

    # Resíduo da Equação de Poisson Reduzida
    rho = ConstantesFisicas.Q_E * (self.N_dop.unsqueeze(-1) - n_3d)
    res_poisson = (
        d2phi_dx2
        - ((phi_pred - Vgs_val) / (self.geom.lambda_g**2))
        + (rho / ConstantesFisicas.EPS_SI)
    )
    loss_pde = torch.mean(res_poisson**2) * 1e-18

    # Condições de Contorno de Dirichlet
    loss_bc_source = (phi_pred[0, 0] - 0.0) ** 2
    loss_bc_drain = (phi_pred[-1, 0] - Vds_val) ** 2
    loss_bc = loss_bc_source + loss_bc_drain

    loss_total = loss_pde + 25.0 * loss_bc
    loss_total.backward()
    self.opt.step()

    return {
        "loss_total": loss_total.item(),
        "loss_pde": loss_pde.item(),
        "loss_bc": loss_bc.item(),
    }


# =============================================================================
# 7. MOTOR DE PROPAGAÇÃO QUÂNTICA DE INCERTEZA (UQ END-TO-END)
# =============================================================================


class PropagadorIncertezaQuantica:
  """Propaga as realizações estocásticas da Bayesian PINN através do solver NEGF:

  phi^(k)(x) -> U^(k)(x) -> G^R,(k) -> T^(k)(E) -> I_D^(k).
  """

  def __init__(
      self,
      pinn: MCDropoutPINN,
      negf: TightBindingNEGF1D,
      geom: GeometriaDispositivo,
      E_grid: torch.Tensor,
  ):
    self.pinn = pinn
    self.negf = negf
    self.geom = geom
    self.E_grid = E_grid

  def avaliar_ponto_operacao(
      self,
      Vgs: float,
      Vds: float,
      n_amostras_mc: int = 40,
  ) -> Dict[str, np.ndarray]:
    """Calcula a distribuição estatística posterior completa no ponto (Vgs, Vds)."""
    N_pts = self.negf.N
    x_norm = (
        torch.linspace(0, 1, N_pts, device=self.negf.device)
        .unsqueeze(-1)
        .to(torch.float32)
    )
    vgs_t = torch.full(
        (N_pts, 1), Vgs, device=self.negf.device, dtype=torch.float32
    )
    vds_t = torch.full(
        (N_pts, 1), Vds, device=self.negf.device, dtype=torch.float32
    )

    # 1. Amostragem de potenciais da MCDropoutPINN: phi^(k)(x)
    amostras_phi = self.pinn.amostrar_potenciais(
        x_norm, vgs_t, vds_t, n_amostras=n_amostras_mc
    )

    correntes_mc = []
    transmissoes_mc = []
    densidades_3d_mc = []

    # 2. Propagação através do NEGF para cada realização
    for k in range(n_amostras_mc):
      phi_k = amostras_phi[k].to(torch.float64)
      U_pot_k = -phi_k  # U^(k)(x) = -q * phi^(k)(x) em eV

      n_1d_k, T_E_k, I_ds_k, _ = self.negf.resolver_transporte(
          U_pot_k, Vds, self.E_grid
      )
      n_3d_k = self.geom.densidade_1d_para_3d(n_1d_k)

      correntes_mc.append(I_ds_k)
      transmissoes_mc.append(T_E_k.cpu().numpy())
      densidades_3d_mc.append(n_3d_k.cpu().numpy())

    arr_phi = amostras_phi.cpu().numpy()
    arr_I = np.array(correntes_mc)
    arr_T = np.array(transmissoes_mc)
    arr_n3d = np.array(densidades_3d_mc)

    return {
        "phi_media": np.mean(arr_phi, axis=0),
        "phi_std": np.std(arr_phi, axis=0),
        "phi_ic95_inf": np.percentile(arr_phi, 2.5, axis=0),
        "phi_ic95_sup": np.percentile(arr_phi, 97.5, axis=0),
        "I_media": float(np.mean(arr_I)),
        "I_std": float(np.std(arr_I)),
        "I_ic95_inf": float(np.percentile(arr_I, 2.5)),
        "I_ic95_sup": float(np.percentile(arr_I, 97.5)),
        "T_media": np.mean(arr_T, axis=0),
        "T_std": np.std(arr_T, axis=0),
        "n3d_media": np.mean(arr_n3d, axis=0),
        "n3d_std": np.std(arr_n3d, axis=0),
        "amostras_corrente": arr_I,
    }

  def varrer_curva_id_vg_com_incerteza(
      self,
      vgs_vetor: np.ndarray,
      Vds_fixo: float = 0.35,
      n_amostras_mc: int = 30,
  ) -> Dict[str, np.ndarray]:
    """Gera a curva de transferência completa I_D(V_GS) com bandas de incerteza bayesianas."""
    medias_I = []
    stds_I = []
    ic_inf_I = []
    ic_sup_I = []

    print(
        f"\n[UQ Transport] Propagando MC Dropout para curva Id-Vg (Vds ="
        f" {Vds_fixo} V)..."
    )
    for vgs in vgs_vetor:
      res_uq = self.avaliar_ponto_operacao(
          float(vgs), Vds_fixo, n_amostras_mc=n_amostras_mc
      )
      medias_I.append(res_uq["I_media"])
      stds_I.append(res_uq["I_std"])
      ic_inf_I.append(res_uq["I_ic95_inf"])
      ic_sup_I.append(res_uq["I_ic95_sup"])
      print(
          f"  Vgs = {vgs:+.2f} V | I_ds = {res_uq['I_media'] * 1e6:8.4f} +-"
          f" {res_uq['I_std'] * 1e6:6.4f} µA (CV:"
          f" {(res_uq['I_std'] / max(res_uq['I_media'], 1e-12)) * 100:5.1f}%)"
      )

    return {
        "vgs": vgs_vetor,
        "I_media": np.array(medias_I),
        "I_std": np.array(stds_I),
        "I_ic_inf": np.array(ic_inf_I),
        "I_ic_sup": np.array(ic_sup_I),
    }


# =============================================================================
# 8. SUÍTE DE EXECUÇÃO MULTIESCALA, BENCHMARK E VISUALIZAÇÃO
# =============================================================================

if __name__ == "__main__":
  torch.manual_seed(42)
  np.random.seed(42)
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 80)
  print(
      " PLATAFORMA INTEGRADA DE NANOELETRÔNICA: NEGF + DD + POISSON + BAYESIAN"
      " PINN"
  )
  print("=" * 80)

  # 1. Configuração Geométrica e Espacial
  geom = GeometriaDispositivo(
      L_canal=15.0e-9, W_largura=5.0e-9, T_corpo=3.0e-9, lambda_gate=2.2e-9
  )
  N_grid = 45
  dx = geom.L / (N_grid - 1)

  # Perfil de Dopagem Tridimensional N+/i/N+ (m⁻³)
  N_dop_3d = torch.zeros(N_grid, dtype=torch.float64, device=dispositivo)
  n_ct = int(0.20 * N_grid)
  N_dop_3d[:n_ct] = 1e26  # Contato de Fonte N+ (1e20 cm⁻³)
  N_dop_3d[-n_ct:] = 1e26  # Contato de Dreno N+
  N_dop_3d[n_ct:-n_ct] = 1e21  # Canal Intrínseco (1e15 cm⁻³)

  # 2. Inicialização dos Solvers Quânticos, Semiclássicos e Eletrostáticos
  E_grid = torch.linspace(
      -0.4, 1.4, 320, dtype=torch.float64, device=dispositivo
  )
  negf_solver = TightBindingNEGF1D(
      N_sites=N_grid, dx=dx, m_eff=0.20, E_F=0.0, T=300.0, device=dispositivo
  )
  dd_solver = DriftDiffusion1D(
      N_sites=N_grid,
      dx=dx,
      geom=geom,
      mu_n=0.04,
      T=300.0,
      device=dispositivo,
  )
  poisson_solver = PoissonReduzido1D(
      N_sites=N_grid, dx=dx, lambda_g=geom.lambda_g, device=dispositivo
  )
  scf_solver = AutoConsistenteSCF(
      negf_solver, poisson_solver, geom, max_iter=80, tol=1e-4, alpha_mix=0.15
  )

  # 3. Solver Auto-Consistente SCF de Referência (Ground Truth Físico)
  print("\n>>> [1/4] Executando Ponto de Operação SCF de Referência...")
  Vgs_op, Vds_op = 0.50, 0.35
  res_scf = scf_solver.executar(Vgs_op, Vds_op, N_dop_3d, E_grid)

  # Transporte Semiclássico sob o mesmo perfil de potencial eletrostático
  n_dd_3d, I_dd_val = dd_solver.resolver_transporte(res_scf["phi"], N_dop_3d)

  print(
      f"  SCF Convergido: {res_scf['convergido']} em {res_scf['iter']} iterações"
  )
  print(
      f"  Corrente Quântica NEGF (Ids):      {res_scf['I_ds'] * 1e6:.4f} µA"
  )
  print(
      f"  Corrente Semiclássica DD (Ids):   {I_dd_val * 1e6:.4f} µA"
  )

  # 4. Treinamento da MCDropoutPINN (Fechamento Quântico-Eletrostático)
  print("\n>>> [2/4] Treinando MCDropoutPINN com Regularização de Monte Carlo...")
  bayesian_pinn = MCDropoutPINN(
      in_dim=3, hidden_dim=48, num_layers=4, p_dropout=0.08
  ).to(dispositivo)
  treinador = TreinadorPINNAutoConsistente(
      bayesian_pinn,
      negf_solver,
      geom,
      N_dop_3d.to(torch.float32),
      E_grid,
      lr=2e-3,
      peso_decay=1e-5,
  )

  for ep in range(1, 151):
    m = treinador.passo_treinamento(Vgs_op, Vds_op)
    if ep % 50 == 0 or ep == 1:
      print(
          f"  Época {ep:03d} | Loss Total: {m['loss_total']:.5e} | PDE:"
          f" {m['loss_pde']:.5e} | BC: {m['loss_bc']:.5e}"
      )

  # 5. Propagação Quântica de Incerteza (UQ End-to-End)
  print("\n>>> [3/4] Propagando Incerteza: phi^(k) -> NEGF -> I_D^(k)...")
  propagador_uq = PropagadorIncertezaQuantica(
      bayesian_pinn, negf_solver, geom, E_grid
  )
  res_uq_ponto = propagador_uq.avaliar_ponto_operacao(
      Vgs_op, Vds_op, n_amostras_mc=50
  )

  print(
      f"  E[I_D] (Média Quântica): {res_uq_ponto['I_media'] * 1e6:.4f} µA"
  )
  print(
      f"  std(I_D) (Incerteza):    {res_uq_ponto['I_std'] * 1e6:.4f} µA"
  )
  print(
      f"  IC 95% Posterior:        [{res_uq_ponto['I_ic95_inf'] * 1e6:.4f} ,"
      f" {res_uq_ponto['I_ic95_sup'] * 1e6:.4f}] µA"
  )

  # 6. Extração da Curva Id-Vg com Incerteza Propagada
  print(
      "\n>>> [4/4] Gerando Curva de Transferência Id-Vg com Bandas Bayesianas..."
  )
  vgs_sweep = np.linspace(-0.1, 0.6, 8)
  res_id_vg_uq = propagador_uq.varrer_curva_id_vg_com_incerteza(
      vgs_sweep, Vds_fixo=Vds_op, n_amostras_mc=30
  )

  # 7. Painel Gráfico Científico Multiescala
  x_nm = np.linspace(0, geom.L * 1e9, N_grid)
  E_ev = E_grid.cpu().numpy()

  fig, axs = plt.subplots(2, 2, figsize=(15, 10))

  # Painel A: Potencial Eletrostático (SCF vs PINN com Incerteza)
  axs[0, 0].plot(
      x_nm,
      res_scf["phi"].cpu().numpy(),
      "k-",
      linewidth=2.2,
      label="SCF Exato",
  )
  axs[0, 0].plot(
      x_nm,
      res_uq_ponto["phi_media"],
      "b--",
      linewidth=2.0,
      label=r"PINN $\mu_\phi(x)$",
  )
  axs[0, 0].fill_between(
      x_nm,
      res_uq_ponto["phi_ic95_inf"],
      res_uq_ponto["phi_ic95_sup"],
      color="blue",
      alpha=0.25,
      label=r"IC 95% Posterior ($\pm 1.96\sigma_\phi$)",
  )
  axs[0, 0].set_title(
      f"Potencial Eletrostático com UQ (Vgs={Vgs_op}V, Vds={Vds_op}V)"
  )
  axs[0, 0].set_xlabel("Posição x (nm)")
  axs[0, 0].set_ylabel("Potencial $\phi(x)$ (V)")
  axs[0, 0].grid(True, linestyle="--", alpha=0.6)
  axs[0, 0].legend()

  # Painel B: Comparação Multiescala de Carga (NEGF Quântico vs Drift-Diffusion Semiclássico)
  axs[0, 1].semilogy(
      x_nm,
      res_scf["n_3d"].cpu().numpy() + 1e18,
      "g-",
      linewidth=2.2,
      label="NEGF Quântico $n_{3D}(x)$",
  )
  axs[0, 1].semilogy(
      x_nm,
      n_dd_3d.cpu().numpy() + 1e18,
      "m--",
      linewidth=2.0,
      label="Drift-Diffusion $n_{3D}(x)$",
  )
  axs[0, 1].semilogy(
      x_nm,
      N_dop_3d.cpu().numpy(),
      "k:",
      alpha=0.7,
      linewidth=1.8,
      label="Dopagem $N_{dop}(x)$",
  )
  axs[0, 1].set_title("Concentração Volumétrica Multiescala [$m^{-3}$]")
  axs[0, 1].set_xlabel("Posição x (nm)")
  axs[0, 1].set_ylabel("Concentração ($m^{-3}$)")
  axs[0, 1].grid(True, linestyle="--", alpha=0.6)
  axs[0, 1].legend()

  # Painel C: Curva de Transferência Id-Vg com Bandas de Incerteza Bayesianas
  axs[1, 0].semilogy(
      res_id_vg_uq["vgs"],
      res_id_vg_uq["I_media"] * 1e6,
      "ro-",
      linewidth=2.0,
      label=r"Corrente Média $E[I_{DS}]$",
  )
  axs[1, 0].fill_between(
      res_id_vg_uq["vgs"],
      np.maximum(res_id_vg_uq["I_ic_inf"] * 1e6, 1e-12),
      res_id_vg_uq["I_ic_sup"] * 1e6,
      color="red",
      alpha=0.25,
      label=r"Banda de Incerteza Bayeasian (IC 95%)",
  )
  axs[1, 0].set_title(
      f"Curva de Transferência $I_D(V_{{GS}})$ com UQ (Vds = {Vds_op} V)"
  )
  axs[1, 0].set_xlabel("Tensão de Porta $V_{GS}$ (V)")
  axs[1, 0].set_ylabel("Corrente de Dreno $I_{DS}$ (µA)")
  axs[1, 0].grid(True, which="both", linestyle="--", alpha=0.6)
  axs[1, 0].legend()

  # Painel D: Densidade Local de Estados Espectral LDOS A(x, E)
  ldos_plot = res_scf["ldos"].cpu().numpy()
  im = axs[1, 1].contourf(
      x_nm, E_ev, ldos_plot, levels=50, cmap="viridis", extend="both"
  )
  axs[1, 1].plot(
      x_nm,
      res_scf["U_x"].cpu().numpy(),
      "w--",
      linewidth=2.0,
      label=r"Perfil de Banda $E_c(x)$",
  )
  axs[1, 1].set_title("Densidade Local de Estados Espectral LDOS $A(x, E)$")
  axs[1, 1].set_xlabel("Posição x (nm)")
  axs[1, 1].set_ylabel("Energia $E$ (eV)")
  axs[1, 1].set_ylim([-0.2, 1.0])
  axs[1, 1].legend(loc="upper right")
  fig.colorbar(im, ax=axs[1, 1], label=r"$A(x, E)$ ($eV^{-1}\cdot m^{-1}$)")

  plt.tight_layout()
  plt.show()
