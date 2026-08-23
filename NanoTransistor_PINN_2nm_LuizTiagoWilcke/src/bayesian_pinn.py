"""Módulo: Bayesian PINN com Propagação Quântica de Incerteza (NEGF & Landauer)

Autor: Luiz Tiago Wilcke
Descrição: Framework integrado para Quantificação de Incerteza (UQ) em Nanoeletrônica:
           1. Bayesian PINN via Monte Carlo Dropout com preservação estrita de
           estado.
           2. Pipeline de propagação quântica estocástica:
              phi^(k)(x) -> U^(k)(x) -> G^R,(k) -> T^(k)(E) -> I_D^(k)
           3. Extração de distribuições preditivas completas:
              - Incerteza espacial no potencial: mu_phi(x) +- sigma_phi(x)
              - Incerteza no transporte: mu_I(Vgs, Vds) +- sigma_I(Vgs, Vds)
              - Bandas de confiança bayesianas nas curvas características
              I_D(V_GS) e I_D(V_DS).
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. CONSTANTES FÍSICAS E ESPECIFICAÇÃO DO DISPOSITIVO
# =============================================================================


class ConstantesFisicas:
  """Constantes fundamentais no SI e conversões de energia."""

  Q_E: float = 1.602176634e-19  # C
  H_PLANCK: float = 6.62607015e-34  # J·s
  H_BAR: float = 1.054571817e-34  # J·s
  M_E: float = 9.1093837e-31  # kg
  K_B_EV: float = 8.617333262e-5  # eV/K
  EPS_0: float = 8.8541878128e-12  # F/m
  EPS_SI: float = 11.7 * 8.8541878128e-12  # Silício (eps_r = 11.7)
  G0: float = 2.0 * (1.602176634e-19**2) / 6.62607015e-34  # ~77.48 µS


class GeometriaDispositivo:
  """Geometria transversal e parâmetros de confinamento."""

  def __init__(
      self,
      L_canal: float = 15.0e-9,  # 15 nm
      W_largura: float = 5.0e-9,  # 5 nm
      T_corpo: float = 3.0e-9,  # 3 nm
      lambda_gate: float = 2.2e-9,  # 2.2 nm
  ):
    self.L = L_canal
    self.W = W_largura
    self.T_body = T_corpo
    self.lambda_g = lambda_gate
    self.A_cross = self.W * self.T_body  # m²

  def densidade_1d_para_3d(self, n_1d: torch.Tensor) -> torch.Tensor:
    """Conversão dimensionalmente consistente de m⁻¹ para m⁻³."""
    return n_1d / self.A_cross


# =============================================================================
# 2. SOLVER TIGHT-BINDING NEGF 1D (RESOLUÇÃO MATRICIAL VETORIZADA)
# =============================================================================


class TightBindingNEGF1D:
  """Solver Quântico 1D baseado em Funções de Green Não-Equilíbrio (NEGF)."""

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
    self.kBT = max(ConstantesFisicas.K_B_EV * T, 1e-7)
    self.device = device

    # Hopping cinético t0 = hbar² / (2 * m* * dx²) em eV
    t0_j = (ConstantesFisicas.H_BAR**2) / (
        2.0 * (self.m_eff * ConstantesFisicas.M_E) * (self.dx**2)
    )
    self.t0 = float(t0_j / ConstantesFisicas.Q_E)

    # Hamiltoniano cinético tridiagonal
    diag_k = 2.0 * self.t0 * torch.ones(self.N, dtype=torch.float64)
    off_k = -self.t0 * torch.ones(self.N - 1, dtype=torch.float64)
    self.H_kin = (
        torch.diag(diag_k) + torch.diag(off_k, 1) + torch.diag(off_k, -1)
    ).to(self.device)

    self.I_mat = torch.eye(
        self.N, dtype=torch.complex128, device=self.device
    )

  def fermi_dirac(
      self, E: torch.Tensor, mu: float
  ) -> torch.Tensor:
    arg = torch.clamp(-(E - mu) / self.kBT, -80.0, 80.0)
    return torch.sigmoid(arg)

  def calcular_sigma_lead(
      self, E_grid: torch.Tensor, U_lead: float
  ) -> torch.Tensor:
    """Autoenergia analítica para contatos semi-infinitos 1D (Open Boundary)."""
    theta = (E_grid - U_lead - 2.0 * self.t0) / (2.0 * self.t0)
    sigma = torch.zeros_like(E_grid, dtype=torch.complex128)

    # Banda de condução
    m_band = torch.abs(theta) <= 1.0
    if m_band.any():
      th = theta[m_band]
      sigma[m_band] = torch.complex(
          self.t0 * th, -self.t0 * torch.sqrt(1.0 - th**2)
      )

    # Estados evanescentes abaixo
    m_bel = theta < -1.0
    if m_bel.any():
      th = theta[m_bel]
      sigma[m_bel] = torch.complex(
          self.t0 * (th + torch.sqrt(th**2 - 1.0)), torch.zeros_like(th)
      )

    # Estados evanescentes acima
    m_abv = theta > 1.0
    if m_abv.any():
      th = theta[m_abv]
      sigma[m_abv] = torch.complex(
          self.t0 * (th - torch.sqrt(th**2 - 1.0)), torch.zeros_like(th)
      )

    return sigma

  def resolver_transporte(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
  ) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Calcula densidade linear n_1D(x) [m⁻¹], Transmissão T(E) e Corrente I_DS [A]."""
    n_E = E_grid.shape[0]
    H_total = self.H_kin + torch.diag(U_potencial.to(torch.float64))
    H_c = H_total.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    sig_S = self.calcular_sigma_lead(E_grid, float(U_potencial[0].item()))
    sig_D = self.calcular_sigma_lead(E_grid, float(U_potencial[-1].item()))

    # Montagem do sistema linear vetorial em lote: A(E) * G^R(E) = I
    A_batch = (
        (E_grid.to(torch.complex128) + 1e-7j).view(-1, 1, 1)
        * self.I_mat.unsqueeze(0)
    ) - H_c.unsqueeze(0)
    A_batch = A_batch.clone()
    A_batch[:, 0, 0] -= sig_S
    A_batch[:, -1, -1] -= sig_D

    I_exp = self.I_mat.unsqueeze(0).expand(n_E, self.N, self.N)
    G_R_batch = torch.linalg.solve(A_batch, I_exp)

    gamma_S = -2.0 * sig_S.imag
    gamma_D = -2.0 * sig_D.imag

    # Transmissão Quântica T(E) = Gamma_S * Gamma_D * |G^R_{0, N-1}|²
    G_R_0N = G_R_batch[:, 0, -1]
    T_E = torch.clamp(
        gamma_S * gamma_D * (G_R_0N.real**2 + G_R_0N.imag**2), min=0.0
    )

    f_S = self.fermi_dirac(E_grid, mu_S)
    f_D = self.fermi_dirac(E_grid, mu_D)

    # Densidade Linear via G^<
    G_R_i0_sq = G_R_batch[:, :, 0].real ** 2 + G_R_batch[:, :, 0].imag ** 2
    G_R_iN_sq = G_R_batch[:, :, -1].real ** 2 + G_R_batch[:, :, -1].imag ** 2
    G_lesser_diag = G_R_i0_sq * (gamma_S * f_S).unsqueeze(
        -1
    ) + G_R_iN_sq * (gamma_D * f_D).unsqueeze(-1)

    dE = (E_grid[1] - E_grid[0]).item()
    integrando_n = (2.0 / (2.0 * np.pi * self.dx)) * G_lesser_diag
    n_1d = torch.sum(integrando_n, dim=0) * dE

    # Corrente Landauer-Büttiker (Amperes)
    integrando_I = T_E * (f_S - f_D)
    I_ds = float(
        ConstantesFisicas.G0 * torch.trapezoid(integrando_I, E_grid).item()
    )

    return n_1d, T_E, I_ds


# =============================================================================
# 3. BAYESIAN PINN (MONTE CARLO DROPOUT COM GERENCIAMENTO DE ESTADO)
# =============================================================================


class BayesianPINN(nn.Module):
  """Rede Neural Informada pela Física Bayesiana com MC Dropout.

  Implementa amostragem estocástica com contexto seguro (no_grad e preservação de estado).
  """

  def __init__(
      self,
      in_dim: int = 3,  # [x_norm, Vgs, Vds]
      hidden_dim: int = 64,
      num_layers: int = 4,
      p_dropout: float = 0.10,
  ):
    super().__init__()
    self.p_dropout = p_dropout
    camadas = []

    camadas.extend(
        [nn.Linear(in_dim, hidden_dim), nn.Tanh(), nn.Dropout(p=p_dropout)]
    )
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
    """Amostra N realizações do potencial phi^(k)(x) preservando o estado do grafo."""
    estado_original = self.training
    self.train()  # Ativa o dropout para amostragem estocástica

    inputs = torch.cat([x_norm, vgs_tensor, vds_tensor], dim=-1)

    with torch.no_grad():
      amostras = torch.stack(
          [self.forward(inputs).squeeze(-1) for _ in range(n_amostras)], dim=0
      )

    self.train(estado_original)  # Restaura o estado original de treinamento
    return amostras  # [n_amostras, N_pontos]


# =============================================================================
# 4. MOTOR DE PROPAGAÇÃO QUÂNTICA DE INCERTEZA (PINN -> NEGF -> UQ EM I_D)
# =============================================================================


class PropagadorIncertezaQuantica:
  """Propaga as realizações estocásticas da Bayesian PINN através do solver NEGF.

  Mapeamento: phi^(k)(x) -> U^(k)(x) -> G^R,(k) -> T^(k)(E) -> I_D^(k)
  """

  def __init__(
      self,
      pinn: BayesianPINN,
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
    """Calcula estatísticas bayesianas completas (Média, Desvio Padrão, Quantis)

    para o potencial, densidade, transmissão e corrente terminal.
    """
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

    # 1. Amostragem de potenciais da Bayesian PINN: phi^(k)(x)
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

      n_1d_k, T_E_k, I_ds_k = self.negf.resolver_transporte(
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
# 5. TREINAMENTO DA BAYESIAN PINN COM RESÍDUO DE POISSON
# =============================================================================


class TreinadorBayesianPINN:
  """Treinamento com regularização L2 (Prior Gaussiano) e resíduo diferencial de Poisson."""

  def __init__(
      self,
      pinn: BayesianPINN,
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
    """Passo de otimização com Dropout ativo durante a diferenciação automática."""
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

    # Carga quântica acoplada via NEGF
    U_quântico = -phi_pred.detach().squeeze(-1).to(torch.float64)
    n_1d, _, _ = self.negf.resolver_transporte(U_quântico, Vds_val, self.E_grid)
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
# 6. EXECUÇÃO, PROPAGAÇÃO DE INCERTEZA E VISUALIZAÇÃO CIENTÍFICA
# =============================================================================

if __name__ == "__main__":
  torch.manual_seed(42)
  np.random.seed(42)
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 80)
  print(
      " BAYESIAN PINN & QUANTUM TRANSPORT: PROPAGAÇÃO DE INCERTEZA NEGF/LANDAUER"
  )
  print("=" * 80)

  # Configuração do Sistema Físico
  geom = GeometriaDispositivo(
      L_canal=15.0e-9, W_largura=5.0e-9, T_corpo=3.0e-9, lambda_gate=2.2e-9
  )
  N_grid = 45
  dx = geom.L / (N_grid - 1)

  # Perfil de Dopagem N+/i/N+
  N_dop_3d = torch.zeros(N_grid, dtype=torch.float64, device=dispositivo)
  n_ct = int(0.20 * N_grid)
  N_dop_3d[:n_ct] = 1e26
  N_dop_3d[-n_ct:] = 1e26
  N_dop_3d[n_ct:-n_ct] = 1e21

  # Grid de Energia e Solvers
  E_grid = torch.linspace(
      -0.4, 1.4, 300, dtype=torch.float64, device=dispositivo
  )
  negf_solver = TightBindingNEGF1D(
      N_sites=N_grid, dx=dx, m_eff=0.20, E_F=0.0, T=300.0, device=dispositivo
  )
  bayesian_pinn = BayesianPINN(
      in_dim=3, hidden_dim=48, num_layers=4, p_dropout=0.08
  ).to(dispositivo)

  treinador = TreinadorBayesianPINN(
      bayesian_pinn,
      negf_solver,
      geom,
      N_dop_3d.to(torch.float32),
      E_grid,
      lr=2e-3,
      peso_decay=1e-5,
  )

  # 1. Treinamento da Bayesian PINN
  print("\n>>> [1/3] Treinando Bayesian PINN com Regularização de Monte Carlo...")
  Vgs_op, Vds_op = 0.50, 0.35
  for ep in range(1, 151):
    m = treinador.passo_treinamento(Vgs_op, Vds_op)
    if ep % 50 == 0 or ep == 1:
      print(
          f"  Época {ep:03d} | Loss Total: {m['loss_total']:.5e} | PDE:"
          f" {m['loss_pde']:.5e} | BC: {m['loss_bc']:.5e}"
      )

  # 2. Propagação Quântica de Incerteza no Ponto de Operação
  print("\n>>> [2/3] Propagando Amostras Estocásticas: phi^(k) -> NEGF -> I_D^(k)...")
  propagador_uq = PropagadorIncertezaQuantica(
      bayesian_pinn, negf_solver, geom, E_grid
  )
  res_ponto = propagador_uq.avaliar_ponto_operacao(
      Vgs_op, Vds_op, n_amostras_mc=50
  )

  print(
      f"  Corrente Média E[I_D]:   {res_ponto['I_media'] * 1e6:.4f} µA"
  )
  print(
      f"  Desvio Padrão std(I_D):  {res_ponto['I_std'] * 1e6:.4f} µA"
  )
  print(
      f"  Intervalo Confiança 95%: [{res_ponto['I_ic95_inf'] * 1e6:.4f} ,"
      f" {res_ponto['I_ic95_sup'] * 1e6:.4f}] µA"
  )

  # 3. Varredura da Curva Id-Vg com Quantificação de Incerteza Completa
  print("\n>>> [3/3] Calculando Curva Id-Vg com Bandas de Incerteza Bayesianas...")
  vgs_sweep = np.linspace(-0.1, 0.6, 8)
  res_id_vg_uq = propagador_uq.varrer_curva_id_vg_com_incerteza(
      vgs_sweep, Vds_fixo=Vds_op, n_amostras_mc=30
  )

  # 4. Geração do Painel Científico com Incerteza Propagada
  x_nm = np.linspace(0, geom.L * 1e9, N_grid)
  E_ev = E_grid.cpu().numpy()

  fig, axs = plt.subplots(2, 2, figsize=(14, 10))

  # Painel A: Potencial Eletrostático com Incerteza Posterior (MC Dropout)
  axs[0, 0].plot(
      x_nm,
      res_ponto["phi_media"],
      "b-",
      linewidth=2.2,
      label=r"Média Preditiva $\mu_\phi(x)$",
  )
  axs[0, 0].fill_between(
      x_nm,
      res_ponto["phi_ic95_inf"],
      res_ponto["phi_ic95_sup"],
      color="blue",
      alpha=0.25,
      label=r"IC 95% Posterior ($\pm 1.96\sigma_\phi$)",
  )
  axs[0, 0].set_title(f"Potencial Eletrostático com UQ (Vgs={Vgs_op}V, Vds={Vds_op}V)")
  axs[0, 0].set_xlabel("Posição no Canal x (nm)")
  axs[0, 0].set_ylabel("Potencial $\phi(x)$ (V)")
  axs[0, 0].grid(True, linestyle="--", alpha=0.6)
  axs[0, 0].legend()

  # Painel B: Espectro de Transmissão Quântica com Incerteza Propagada
  axs[0, 1].plot(
      E_ev,
      res_ponto["T_media"],
      "m-",
      linewidth=2.0,
      label=r"Transmissão Média $\mu_T(E)$",
  )
  axs[0, 1].fill_between(
      E_ev,
      np.maximum(res_ponto["T_media"] - 1.96 * res_ponto["T_std"], 0.0),
      res_ponto["T_media"] + 1.96 * res_ponto["T_std"],
      color="purple",
      alpha=0.25,
      label=r"Incerteza Propagada em $T(E)$ (IC 95%)",
  )
  axs[0, 1].set_title("Transmissão Quântica $T(E)$ com UQ Propagada")
  axs[0, 1].set_xlabel("Energia $E$ (eV)")
  axs[0, 1].set_ylabel("Transmissão $T(E)$")
  axs[0, 1].grid(True, linestyle="--", alpha=0.6)
  axs[0, 1].legend()

  # Painel C: Curva de Transferência Id-Vg com Bandas de Confiança Bayesianas
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

  # Painel D: Coeficiente de Variação Relativa da Corrente (sigma_I / mu_I)
  cv_corrente = (res_id_vg_uq["I_std"] / res_id_vg_uq["I_media"]) * 100.0
  axs[1, 1].plot(
      res_id_vg_uq["vgs"],
      cv_corrente,
      "k-s",
      linewidth=2.0,
      markersize=6,
      label="Incerteza Relativa da Corrente (%)",
  )
  axs[1, 1].set_title(r"Incerteza Relativa $CV = \sigma_I / \mu_I$ vs $V_{GS}$")
  axs[1, 1].set_xlabel("Tensão de Porta $V_{GS}$ (V)")
  axs[1, 1].set_ylabel("Incerteza Relativa (%)")
  axs[1, 1].grid(True, linestyle="--", alpha=0.6)
  axs[1, 1].legend()

  plt.tight_layout()
  plt.show()
