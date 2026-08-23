"""Framework Quântico-Eletrostático Unificado (NEGF 1D, Poisson Reduzido & PINN)

Autor: Luiz Tiago Wilcke
Descrição: Solver completo para nanotransistores (DG-MOSFET/Nanowire).
           - Autoenergias analíticas de contatos semi-infinitos 1D (Open Boundary Conditions).
           - Solver matricial Tight-Binding NEGF com resolução via torch.linalg.solve.
           - Equação de Poisson Reduzida com acoplamento capacitivo de porta e consistência dimensional [m⁻³].
           - Loop Auto-Consistente (SCF) exato de referência com amortecimento de Picard.
           - PINN com fechamento do ciclo físico U(x) -> NEGF -> n_3D(x) -> Poisson Residual.
           - Rotinas completas de caracterização: Id-Vg, Id-Vd, Subthreshold Swing (SS) e LDOS espectral.
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. CONSTANTES FÍSICAS UNIVERSAIS E ESTRUTURA GEOMÉTRICA
# =============================================================================


class ConstantesFisicas:
  """Constantes fundamentais no Sistema Internacional (SI) e conversões."""

  Q_E: float = 1.602176634e-19  # C (Carga elementar)
  H_PLANCK: float = 6.62607015e-34  # J·s (Constante de Planck)
  H_BAR: float = 1.054571817e-34  # J·s (Planck reduzida)
  M_E: float = 9.1093837e-31  # kg (Massa do elétron livre)
  K_B_EV: float = 8.617333262e-5  # eV/K (Constante de Boltzmann)
  EPS_0: float = 8.8541878128e-12  # F/m (Permissividade do vácuo)
  EPS_SI: float = (
      11.7 * 8.8541878128e-12
  )  # F/m (Permissividade do Silício: eps_r = 11.7)
  G0: float = (
      2.0 * (1.602176634e-19**2) / 6.62607015e-34
  )  # 2q²/h ≈ 7.74809e-5 S (A/eV)


class GeometriaDispositivo:
  """Parâmetros geométricos do transistor para consistência dimensional estrita."""

  def __init__(
      self,
      L_canal: float = 15.0e-9,  # Comprimento do canal (m) -> 15 nm
      W_largura: float = 5.0e-9,  # Largura do canal (m) -> 5 nm
      T_corpo: float = 3.0e-9,  # Espessura do corpo semicondutor (m) -> 3 nm
      lambda_gate: float = 2.2e-9,  # Comprimento de triagem eletrostática da porta (m)
  ):
    self.L = L_canal
    self.W = W_largura
    self.T_body = T_corpo
    self.lambda_g = lambda_gate
    # Área da seção transversal transversal ortogonal ao transporte: A = W * T_body (m²)
    self.A_cross = self.W * self.T_body

  def densidade_1d_para_3d(self, n_1d: torch.Tensor) -> torch.Tensor:
    """Converte densidade linear [m⁻¹] para volumétrica [m⁻³] de forma dimensionalmente exata."""
    return n_1d / self.A_cross


# =============================================================================
# 2. AUTOENERGIAS ANALÍTICAS DE CONTATO (OPEN BOUNDARY CONDITIONS 1D)
# =============================================================================


class AutoEnergiaLead1D:
  """Calcula a autoenergia analítica exata Sigma_lead^R(E) para contatos semi-infinitos 1D.

  Relação de dispersão no contato: E - U_lead - 2t0 = -2t0 * cos(k*a).
  """

  @staticmethod
  def calcular_sigma(
      E_grid: torch.Tensor, U_lead: float, t0: float
  ) -> torch.Tensor:
    """Retorna o vetor complexo Sigma^R(E) para todas as energias do grid."""
    theta = (E_grid - U_lead - 2.0 * t0) / (2.0 * t0)
    sigma = torch.zeros_like(E_grid, dtype=torch.complex128)

    # 1. Estados propagantes dentro da banda de condução (|theta| <= 1)
    mask_band = torch.abs(theta) <= 1.0
    if mask_band.any():
      th_b = theta[mask_band]
      # Onda progressiva para fora do canal: Sigma = -t0 * exp(i*k*a) = t0 * theta - i * t0 * sqrt(1 - theta²)
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

  Evita inversão explícita inv(A) utilizando torch.linalg.solve(A, B).
  """

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

    # Energia de acoplamento cinético de hopping: t0 = hbar² / (2 * m* * dx²) em eV
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
    """Distribuição estatística de Fermi-Dirac protegida contra saturação de float."""
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def resolver_transporte(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, float, torch.Tensor]:
    """Calcula densidade linear n_1D(x) [m⁻¹], espectro T(E), corrente terminal I_DS [A]

    e a Densidade Local de Estados Espectral A(x, E) [eV⁻¹·m⁻¹].
    """
    n_energies = E_grid.shape[0]
    H_total = self.H_kin + torch.diag(U_potencial.to(torch.float64))
    H_c = H_total.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    U_S = float(U_potencial[0].item())
    U_D = float(U_potencial[-1].item())

    # Pré-cálculo das autoenergias dos contatos
    if self.modo_contato == "rigoroso":
      sigma_S_vec = AutoEnergiaLead1D.calcular_sigma(E_grid, U_S, self.t0)
      sigma_D_vec = AutoEnergiaLead1D.calcular_sigma(E_grid, U_D, self.t0)
    else:
      sigma_S_vec = torch.full(
          (n_energies,), -1j * (self.gamma_wbl / 2.0), dtype=torch.complex128
      )
      sigma_D_vec = torch.full(
          (n_energies,), -1j * (self.gamma_wbl / 2.0), dtype=torch.complex128
      )

    # Montagem do tensor de sistemas lineares em lote: A_batch [n_energies, N, N]
    A_batch = (
        (E_grid.to(torch.complex128) + 1e-7j).view(-1, 1, 1)
        * self.I_mat.unsqueeze(0)
    ) - H_c.unsqueeze(0)
    A_batch = A_batch.clone()
    A_batch[:, 0, 0] -= sigma_S_vec
    A_batch[:, -1, -1] -= sigma_D_vec

    # Resolução exata de G^R(E): (A_batch) @ G^R = I -> G^R = solve(A_batch, I)
    I_expanded = self.I_mat.unsqueeze(0).expand(n_energies, self.N, self.N)
    G_R_batch = torch.linalg.solve(A_batch, I_expanded)  # [n_energies, N, N]
    G_A_batch = G_R_batch.conj().transpose(-2, -1)

    # Matrizes de acoplamento: Gamma = i * (Sigma - Sigma^dagger)
    gamma_S_val = -2.0 * sigma_S_vec.imag  # [n_energies]
    gamma_D_val = -2.0 * sigma_D_vec.imag

    # Transmissão Quântica T(E) = Tr[Gamma_S @ G^R @ Gamma_D @ G^A]
    # Usando a propriedade: T(E) = Gamma_S * Gamma_D * |G^R[0, N-1]|²
    G_R_0N = G_R_batch[:, 0, -1]
    T_E = gamma_S_val * gamma_D_val * (G_R_0N.real**2 + G_R_0N.imag**2)
    T_E = torch.clamp(T_E.real, min=0.0)

    # Distribuições de Fermi nos contatos
    f_S = self.fermi_dirac(E_grid, mu_S)
    f_D = self.fermi_dirac(E_grid, mu_D)

    # Função de Green Menor: G^<(E) = G^R @ (Gamma_S * f_S + Gamma_D * f_D) @ G^A
    # Diagonal de G^<: G^<_{ii} = |G^R_{i,0}|² * (Gamma_S * f_S) + |G^R_{i,N-1}|² * (Gamma_D * f_D)
    G_R_i0_sq = (
        G_R_batch[:, :, 0].real ** 2 + G_R_batch[:, :, 0].imag ** 2
    )  # [n_E, N]
    G_R_iN_sq = (
        G_R_batch[:, :, -1].real ** 2 + G_R_batch[:, :, -1].imag ** 2
    )  # [n_E, N]

    G_lesser_diag = G_R_i0_sq * (gamma_S_val * f_S).unsqueeze(
        -1
    ) + G_R_iN_sq * (gamma_D_val * f_D).unsqueeze(-1)

    # Densidade de Estados Local (LDOS): A(x, E) = - (1/pi) * Im[diag(G^R)] / dx
    ldos = -(1.0 / np.pi) * (G_R_batch.diagonal(dim1=-2, dim2=-1).imag) / self.dx

    # Densidade Linear n_1D(x) integrada em energia com degenerescência gs=2, gv=1:
    # n_1D(x) = (2 / (2*pi * dx)) * int G^<_{ii}(E) dE
    dE = (E_grid[1] - E_grid[0]).item()
    integrando_n = (2.0 / (2.0 * np.pi * self.dx)) * G_lesser_diag  # [n_E, N]
    n_1d = torch.sum(integrando_n, dim=0) * dE  # [N]

    # Corrente Landauer-Büttiker Terminal (Amperes)
    integrando_I = T_E * (f_S - f_D)
    I_ds = float(
        ConstantesFisicas.G0 * torch.trapezoid(integrando_I, E_grid).item()
    )

    return n_1d, T_E, I_ds, ldos


# =============================================================================
# 4. EQUAÇÃO DE POISSON REDUZIDA COM ACOPLAMENTO DE PORTA (THIN-BODY)
# =============================================================================


class PoissonReduzido1D:
  """Modelo Eletrostático de Poisson 1D com triagem de porta para DG-MOSFET / Nanowire:

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

    # Operador linear tridiagonal: d²/dx² - 1/lambda_g²
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
    """Resolve o potencial eletrostático linear phi(x) em Volts via Dirichlet BCs."""
    rho = ConstantesFisicas.Q_E * (N_dop_3d - n_3d)
    RHS = -(rho / ConstantesFisicas.EPS_SI) - (Vgs / self.lambda_sq)

    A = self.M_poisson.clone()
    b = RHS.clone()

    # Condições de Contorno de Dirichlet nos contatos de Fonte e Dreno
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = Vs

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = Vd

    return torch.linalg.solve(A, b)


# =============================================================================
# 5. LOOP AUTO-CONSISTENTE POISSON-NEGF (SCF BENCHMARK EXATO)
# =============================================================================


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
    """Executa o ciclo auto-consistente: phi -> U(x) -> NEGF -> n_3D -> Poisson -> phi_novo."""
    # Chute inicial linear
    phi = torch.linspace(
        0.0, Vds, self.negf.N, dtype=torch.float64, device=self.negf.device
    )
    n_3d = torch.zeros_like(N_dop_3d)

    for it in range(self.max_iter):
      # Energia Potencial Eletrostática: U(x) = -q * phi(x) em eV
      U_pot = -phi

      # 1. Transporte Quântico NEGF
      n_1d, T_E, I_ds, ldos = self.negf.resolver_transporte(U_pot, Vds, E_grid)
      n_3d_novo = self.geom.densidade_1d_para_3d(n_1d)

      # 2. Eletrostática de Poisson
      phi_novo = self.poisson.resolver(n_3d_novo, N_dop_3d, Vgs, 0.0, Vds)

      # 3. Critério de Convergência e Amortecimento
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
# 6. PHYSICS-INFORMED NEURAL NETWORK (PINN PARAMÉTRICA COM CICLO FECHADO)
# =============================================================================


class TransistorPINN(nn.Module):
  """Surrogate Paramétrico Contínuo: (x_norm, Vgs, Vds) -> phi(x, Vgs, Vds)."""

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
  """Treinador com acoplamento quântico fechado: phi_pred -> U(x) -> NEGF -> n_3D -> Resíduo PDE."""

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

  def passo_treinamento(self, Vgs_val: float, Vds_val: float) -> Dict[str, float]:
    """Passo de otimização fechando o resíduo de Poisson na densidade NEGF real."""
    self.opt.zero_grad()

    N_pts = self.negf.N
    x_real = torch.linspace(
        0, self.geom.L, N_pts, dtype=torch.float32, requires_grad=True
    ).unsqueeze(-1)
    x_norm = x_real / self.geom.L
    vgs_t = torch.full((N_pts, 1), Vgs_val, dtype=torch.float32)
    vds_t = torch.full((N_pts, 1), Vds_val, dtype=torch.float32)

    # 1. Inferência da PINN
    phi_pred = self.pinn(x_norm, vgs_t, vds_t)

    # 2. Diferenciação Automática Exata (Autograd)
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

    # 3. Fechamento de Ciclo Quântico com NEGF
    U_quântico = -phi_pred.detach().squeeze(-1).to(torch.float64)
    n_1d, _, _, _ = self.negf.resolver_transporte(
        U_quântico, Vds_val, self.E_grid
    )
    n_3d_quântico = (
        self.geom.densidade_1d_para_3d(n_1d).unsqueeze(-1).to(torch.float32)
    )

    # 4. Resíduo da Equação de Poisson Reduzida
    rho = ConstantesFisicas.Q_E * (self.N_dop.unsqueeze(-1) - n_3d_quântico)
    res_poisson = (
        d2phi_dx2
        - ((phi_pred - Vgs_val) / (self.geom.lambda_g**2))
        + (rho / ConstantesFisicas.EPS_SI)
    )
    loss_pde = torch.mean(res_poisson**2) * 1e-18  # Normalização de escala

    # 5. Perdas de Contorno nos Eletrodos (Dirichlet)
    loss_bc_source = (phi_pred[0, 0] - 0.0) ** 2
    loss_bc_drain = (phi_pred[-1, 0] - Vds_val) ** 2
    loss_bc = loss_bc_source + loss_bc_drain

    # 6. Atualização dos Pesos
    loss_total = loss_pde + 25.0 * loss_bc
    loss_total.backward()
    self.opt.step()

    return {
        "loss_total": loss_total.item(),
        "loss_pde": loss_pde.item(),
        "loss_bc": loss_bc.item(),
    }


# =============================================================================
# 7. ROTINAS DE CARACTERIZAÇÃO DO DISPOSITIVO (Id-Vg, Id-Vd, SS & Ion/Ioff)
# =============================================================================


class CaracterizacaoDispositivo:
  """Extração de métricas elétricas e curvas de transporte de nanoeletrônica."""

  @staticmethod
  def varrer_transferencia_id_vg(
      scf_solver: AutoConsistenteSCF,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
      Vds_fixo: float = 0.40,
      vgs_range: np.ndarray = np.linspace(-0.1, 0.7, 9),
  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Calcula a curva Id-Vg, extrai o Subthreshold Swing (SS em mV/dec) e a razão Ion/Ioff."""
    correntes = []
    print(f"\n[Caracterização] Varrimento Id-Vg (Vds = {Vds_fixo:.2f} V)...")

    for vg in vgs_range:
      res = scf_solver.executar(float(vg), Vds_fixo, N_dop_3d, E_grid)
      correntes.append(res["I_ds"])
      print(f"  Vgs = {vg:+.2f} V | I_ds = {res['I_ds'] * 1e6:8.4f} µA")

    ids_arr = np.array(correntes)
    ids_arr = np.maximum(ids_arr, 1e-14)  # Piso numérico para log10

    # Extração de Subthreshold Swing: SS = d(Vgs) / d(log10(Ids)) mínimo no subsetor sublinear
    log_ids = np.log10(ids_arr)
    d_log_ids = np.gradient(log_ids, vgs_range)
    ss_values = 1000.0 / np.maximum(
        d_log_ids, 1e-3
    )  # mV/década (ideal ~60 mV/dec a 300K)
    ss_min = float(
        np.min(ss_values[(vgs_range >= 0.0) & (vgs_range <= 0.35)])
    )

    ion = float(ids_arr[-1])
    ioff = float(ids_arr[0])
    ion_ioff = ion / ioff

    return vgs_range, ids_arr, ss_min, ion_ioff

  @staticmethod
  def varrer_saida_id_vd(
      scf_solver: AutoConsistenteSCF,
      N_dop_3d: torch.Tensor,
      E_grid: torch.Tensor,
      vgs_valores: List[float] = [0.2, 0.4, 0.6],
      vds_range: np.ndarray = np.linspace(0.02, 0.50, 7),
  ) -> Dict[float, np.ndarray]:
    """Calcula a família de curvas de saída Id-Vd para múltiplos valores de porta."""
    curvas_saida = {}
    print("\n[Caracterização] Família de Curvas de Saída Id-Vd...")

    for vg in vgs_valores:
      i_lista = []
      for vd in vds_range:
        res = scf_solver.executar(vg, float(vd), N_dop_3d, E_grid)
        i_lista.append(res["I_ds"])
      curvas_saida[vg] = np.array(i_lista)
      print(f"  Vgs = {vg:.2f} V concluído. I_sat ≈ {i_lista[-1] * 1e6:.3f} µA")

    return curvas_saida


# =============================================================================
# 8. DEMONSTRAÇÃO EXECUTÁVEL COMPLETA E GERAÇÃO DE RESULTADOS
# =============================================================================

if __name__ == "__main__":
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print("=" * 80)
  print(
      " PLATAFORMA QUÂNTICO-ELETROSTÁTICA COMPLETA: NEGF 1D + POISSON + PINN"
  )
  print("=" * 80)

  # 1. Configuração do Dispositivo (DG-MOSFET 15 nm)
  geom = GeometriaDispositivo(
      L_canal=15.0e-9, W_largura=5.0e-9, T_corpo=3.0e-9, lambda_gate=2.2e-9
  )
  N_grid = 45
  dx = geom.L / (N_grid - 1)

  # Perfil Físico de Dopagem Tridimensional N+/i/N+ (m⁻³)
  N_dop_3d = torch.zeros(N_grid, dtype=torch.float64, device=dispositivo)
  n_contato = int(0.20 * N_grid)
  N_dop_3d[:n_contato] = 1e26  # Fonte dopada N+ (1e20 cm⁻³)
  N_dop_3d[-n_contato:] = 1e26  # Dreno dopado N+
  N_dop_3d[n_contato:-n_contato] = 1e21  # Canal central intrínseco (1e15 cm⁻³)

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
      negf, poisson, geom, max_iter=80, tol=1e-4, alpha_mix=0.15
  )

  # Grid de Energia de Integração (em eV)
  E_grid = torch.linspace(
      -0.4, 1.4, 350, dtype=torch.float64, device=dispositivo
  )

  # --- PARTE A: Solução de Referência SCF (Ponto de Operação) ---
  print("\n>>> [1/4] Executando Ponto de Operação SCF de Referência...")
  Vgs_op, Vds_op = 0.50, 0.30
  res_scf = scf.executar(Vgs_op, Vds_op, N_dop_3d, E_grid)

  print(
      f"  SCF Convergido: {res_scf['convergido']} em {res_scf['iter']} iterações"
  )
  print(f"  Corrente Landauer SCF (Ids): {res_scf['I_ds'] * 1e6:.4f} µA")
  print(
      f"  Densidade 3D Pico no Canal:  {torch.max(res_scf['n_3d']).item():.3e}"
      " m⁻³"
  )

  # --- PARTE B: Treinamento da PINN Fechada com NEGF ---
  print("\n>>> [2/4] Treinando PINN com Acoplamento Quântico Fechado...")
  pinn_model = TransistorPINN(hidden_dim=48, num_layers=4).to(dispositivo)
  treinador = TreinadorPINNAutoConsistente(
      pinn_model,
      negf,
      geom,
      N_dop_3d.to(torch.float32),
      E_grid,
      lr=1.5e-3,
  )

  for ep in range(1, 151):
    metricas = treinador.passo_treinamento(Vgs_op, Vds_op)
    if ep % 50 == 0 or ep == 1:
      print(
          f"  Época {ep:03d} | Loss Total: {metricas['loss_total']:.5e} | PDE:"
          f" {metricas['loss_pde']:.5e} | BC: {metricas['loss_bc']:.5e}"
      )

  # Avaliação da PINN
  x_eval = (
      torch.linspace(0, 1, N_grid, device=dispositivo)
      .unsqueeze(-1)
      .to(torch.float32)
  )
  vgs_eval = torch.full((N_grid, 1), Vgs_op, device=dispositivo, dtype=torch.float32)
  vds_eval = torch.full((N_grid, 1), Vds_op, device=dispositivo, dtype=torch.float32)

  with torch.no_grad():
    phi_pinn = pinn_model(x_eval, vgs_eval, vds_eval).cpu().numpy().squeeze()

  # --- PARTE C: Caracterização Elétrica (Id-Vg e Id-Vd) ---
  print("\n>>> [3/4] Extraindo Características Globais do Nanotransistor...")
  vg_sweep, ids_transfer, ss_sub, ion_ioff_ratio = (
      CaracterizacaoDispositivo.varrer_transferencia_id_vg(
          scf,
          N_dop_3d,
          E_grid,
          Vds_fixo=0.35,
          vgs_range=np.linspace(-0.1, 0.6, 8),
      )
  )

  vd_sweep = np.linspace(0.02, 0.45, 6)
  curvas_saida = CaracterizacaoDispositivo.varrer_saida_id_vd(
      scf,
      N_dop_3d,
      E_grid,
      vgs_valores=[0.2, 0.4, 0.6],
      vds_range=vd_sweep,
  )

  print(f"\n  [Métricas de Nanoeletrônica]")
  print(f"  Subthreshold Swing (SS): {ss_sub:.1f} mV/década")
  print(f"  Razão Ion/Ioff:          {ion_ioff_ratio:.2e}")

  # --- PARTE D: Painel Gráfico Científico ---
  print("\n>>> [4/4] Gerando Painel de Resultados Físicos...")
  x_nm = np.linspace(0, geom.L * 1e9, N_grid)

  fig, axs = plt.subplots(2, 2, figsize=(14, 10))

  # 1. Comparação de Potencial Eletrostático (SCF vs PINN)
  axs[0, 0].plot(
      x_nm,
      res_scf["phi"].cpu().numpy(),
      "b-",
      linewidth=2.2,
      label="SCF Exato",
  )
  axs[0, 0].plot(
      x_nm,
      phi_pinn,
      "r--",
      linewidth=2.0,
      label="PINN Fechada",
  )
  axs[0, 0].set_title(f"Potencial phi(x) (Vgs={Vgs_op}V, Vds={Vds_op}V)")
  axs[0, 0].set_xlabel("Posição x (nm)")
  axs[0, 0].set_ylabel("Potencial (V)")
  axs[0, 0].grid(True, linestyle="--", alpha=0.6)
  axs[0, 0].legend()

  # 2. Densidade de Carga e Dopagem 3D
  axs[0, 1].semilogy(
      x_nm,
      res_scf["n_3d"].cpu().numpy() + 1e18,
      "g-",
      linewidth=2.0,
      label="Densidade Quântica n_3D(x)",
  )
  axs[0, 1].semilogy(
      x_nm,
      N_dop_3d.cpu().numpy(),
      "k:",
      alpha=0.7,
      linewidth=1.8,
      label="Dopagem N_dop(x)",
  )
  axs[0, 1].set_title("Concentração Volumétrica de Portadores [m⁻³]")
  axs[0, 1].set_xlabel("Posição x (nm)")
  axs[0, 1].set_ylabel("Concentração (m⁻³)")
  axs[0, 1].grid(True, linestyle="--", alpha=0.6)
  axs[0, 1].legend()

  # 3. Curva de Transferência Id-Vg (Log scale)
  axs[1, 0].semilogy(
      vg_sweep, ids_transfer * 1e6, "ro-", linewidth=2.0, markersize=5
  )
  axs[1, 0].set_title(
      f"Curva de Transferência Id-Vg (SS = {ss_sub:.1f} mV/dec)"
  )
  axs[1, 0].set_xlabel("Tensão de Porta Vgs (V)")
  axs[1, 0].set_ylabel("Corrente de Dreno I_ds (µA)")
  axs[1, 0].grid(True, which="both", linestyle="--", alpha=0.6)

  # 4. Família de Curvas de Saída Id-Vd
  for vg_val, ids_vd in curvas_saida.items():
    axs[1, 1].plot(
        vd_sweep,
        ids_vd * 1e6,
        marker="s",
        linewidth=2.0,
        label=f"Vgs = {vg_val:.1f} V",
    )
  axs[1, 1].set_title("Família de Curvas de Saída Id-Vd")
  axs[1, 1].set_xlabel("Tensão Dreno-Fonte Vds (V)")
  axs[1, 1].set_ylabel("Corrente de Dreno I_ds (µA)")
  axs[1, 1].grid(True, linestyle="--", alpha=0.6)
  axs[1, 1].legend()

  plt.tight_layout()
  plt.show()
