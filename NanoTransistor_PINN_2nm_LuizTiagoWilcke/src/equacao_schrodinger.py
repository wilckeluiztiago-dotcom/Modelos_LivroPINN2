"""Módulo: Solver de Schrödinger 1D com PINN (Massa Efetiva, Auto-Consistência e Normalização)

Autor: Luiz Tiago Wilcke
Descrição: Solver de autovalores de Schrödinger 1D para poços quânticos e canais ultra-finos (UTBB/Nanosheet):
           1. Adimensionalização física rigorosa com energia de confinamento E_conf e potencial térmico V_T.
           2. Cálculo de derivadas de 2ª ordem exatas via torch.autograd (sem diferenças finitas).
           3. Perda composta com quadratura numérica (torch.trapezoid), condições de contorno de Dirichlet
              e penalização de ortogonalidade para múltiplos estados excitados.
           4. Acoplamento auto-consistente Schrödinger-Poisson: V(x) -> psi_n, E_n -> n_3D(x) -> Poisson -> V(x).
           5. Validação analítica contra o poço quântico infinito (solução exata).
"""

from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. PARÂMETROS MATERIAIS E CONSTANTES FÍSICAS (SILÍCIO / SI)
# =============================================================================


class ParametrosSilicio:
  """Constantes fundamentais e propriedades do Silício."""

  Q_E: float = 1.602176634e-19  # C
  H_PLANCK: float = 6.62607015e-34  # J·s
  H_BAR: float = 1.054571817e-34  # J·s
  M_E: float = 9.1093837e-31  # kg
  K_B: float = 1.380649e-23  # J/K
  K_B_EV: float = 8.617333262e-5  # eV/K
  EPS_0: float = 8.8541878128e-12  # F/m
  EPS_SI: float = 11.7 * 8.8541878128e-12  # F/m

  def __init__(self, T: float = 300.0):
    self.T = T
    self.q = self.Q_E
    self.hbar = self.H_BAR
    self.m0 = self.M_E
    self.eps_si = self.EPS_SI
    # Massas efetivas no Silício (Vale de condução Delta)
    self.m_l = 0.98  # Longitudinal
    self.m_t = 0.19  # Transversal
    self.m_eff_conf = 0.98  # Massa de confinamento na direção [100] (vales não-degenerados)
    self.m_dos_2d = 2.0 * np.sqrt(
        self.m_t * self.m_l
    )  # Massa DOS para sub-bandas 2D

  def VT(self, T: Optional[float] = None) -> float:
    """Tensão térmica V_T = k_B * T / q em Volts."""
    temp = T if T is not None else self.T
    return (self.K_B * temp) / self.Q_E

  def massa_efetiva_kg(self) -> float:
    """Retorna a massa efetiva de confinamento em kg."""
    return self.m_eff_conf * self.m0


# =============================================================================
# 2. RESIDUAL FÍSICO DE SCHRÖDINGER E REGULARIZAÇÕES
# =============================================================================


class ResidualSchrodinger:
  """Calcula o resíduo diferencial de Schrödinger 1D adimensionalizado,

  a quadratura de normalização integral e as perdas de contorno/ortogonalidade.
  """

  def __init__(
      self,
      mat: Optional[ParametrosSilicio] = None,
      t_nm: float = 2.0,  # Espessura do canal quântico (2 nm)
      T: float = 300.0,
      device: torch.device = torch.device("cpu"),
  ):
    self.mat = mat or ParametrosSilicio(T=T)
    self.t = t_nm * 1e-9  # metros
    self.T = T
    self.device = device
    self.q = self.mat.q
    self.hbar = self.mat.hbar
    self.m_star = self.mat.massa_efetiva_kg()
    self.VT_val = self.mat.VT(T)

    # Energia de confinamento base: E_conf = hbar² / (2 * m* * t²) em Joules
    self.E_conf = (self.hbar**2) / (2.0 * self.m_star * (self.t**2))
    self.E_conf_eV = self.E_conf / self.q

    # Coeficiente adimensional da derivada de segunda ordem:
    # alpha = E_conf / (q * V_T) = E_conf_eV / V_T
    self.alpha = self.E_conf_eV / self.VT_val

  def residual(
      self,
      psi: torch.Tensor,
      E_star: torch.Tensor,
      V_star: torch.Tensor,
      y_star: torch.Tensor,
  ) -> torch.Tensor:
    """Forma adimensionalizada de Schrödinger:

    -alpha * (d²psi / dy*²) + (V*(y*) - E*) * psi = 0
    y* ∈ [0, 1], V* = V / V_T, E* = E / V_T.
    """
    dpsi = torch.autograd.grad(
        psi,
        y_star,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
    )[0]

    d2psi = torch.autograd.grad(
        dpsi,
        y_star,
        grad_outputs=torch.ones_like(dpsi),
        create_graph=True,
        retain_graph=True,
    )[0]

    return -self.alpha * d2psi + (V_star - E_star) * psi

  def perda_normalizacao(
      self, psi: torch.Tensor, y_star: torch.Tensor
  ) -> torch.Tensor:
    """Calcula a violação da integral de normalização quântica: (∫ |psi|² dy* - 1)² via regra do trapézio."""
    integral_norm = torch.trapezoid(psi.squeeze() ** 2, y_star.squeeze())
    return (integral_norm - 1.0) ** 2

  def perda_contorno(self, psi: torch.Tensor) -> torch.Tensor:
    """Impõe condições de contorno de Dirichlet nas paredes do poço: psi(0) = 0 e psi(1) = 0."""
    psi_esq = psi[0]
    psi_dir = psi[-1]
    return psi_esq**2 + psi_dir**2

  def perda_ortogonalidade(
      self,
      psi_n: torch.Tensor,
      estados_anteriores: List[torch.Tensor],
      y_star: torch.Tensor,
  ) -> torch.Tensor:
    """Garante a ortogonalidade estrita com estados de menor energia: (∫ psi_n * psi_k dy*)² = 0."""
    if not estados_anteriores:
      return torch.tensor(0.0, device=self.device, dtype=y_star.dtype)

    loss_ortho = torch.tensor(0.0, device=self.device, dtype=y_star.dtype)
    y_flat = y_star.squeeze()
    psi_n_flat = psi_n.squeeze()

    for psi_k in estados_anteriores:
      psi_k_flat = psi_k.detach().squeeze()
      sobreposicao = torch.trapezoid(psi_n_flat * psi_k_flat, y_flat)
      loss_ortho += sobreposicao**2

    return loss_ortho

  def calcular_energia_rayleigh(
      self,
      psi: torch.Tensor,
      V_star: torch.Tensor,
      y_star: torch.Tensor,
  ) -> torch.Tensor:
    """Calcula o autovalor de energia variacional via Quociente de Rayleigh:

    <E*> = ∫ [ alpha*(dpsi/dy*)² + V*|psi|² ] dy* / ∫ |psi|² dy*
    """
    dpsi = torch.autograd.grad(
        psi,
        y_star,
        grad_outputs=torch.ones_like(psi),
        create_graph=True,
        retain_graph=True,
    )[0]

    integrando_num = (
        self.alpha * (dpsi.squeeze() ** 2) + V_star.squeeze() * (psi.squeeze() ** 2)
    )
    integrando_den = psi.squeeze() ** 2

    y_flat = y_star.squeeze()
    num = torch.trapezoid(integrando_num, y_flat)
    den = torch.trapezoid(integrando_den, y_flat)

    return num / (den + 1e-12)


# =============================================================================
# 3. REDE NEURAL INFORMADA PELA FÍSICA (SCHRÖDINGER PINN)
# =============================================================================


class SchrodingerPINN(nn.Module):
  """Rede Neural com autovalor de energia treinável (E*) para resolver autofunções e autovalores."""

  def __init__(
      self,
      hidden_dim: int = 64,
      num_layers: int = 3,
      E_star_inicial: float = 1.0,
      forcar_contorno_ansatz: bool = False,
  ):
    super().__init__()
    self.forcar_ansatz = forcar_contorno_ansatz

    camadas = [nn.Linear(1, hidden_dim), nn.Tanh()]
    for _ in range(num_layers - 1):
      camadas.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
    camadas.append(nn.Linear(hidden_dim, 1))

    self.rede = nn.Sequential(*camadas)
    # Autovalor adimensional E* tratado como parâmetro variacional treinável
    self.E_star = nn.Parameter(
        torch.tensor(float(E_star_inicial), dtype=torch.float64)
    )

  def forward(self, y_star: torch.Tensor) -> torch.Tensor:
    """Retorna psi(y*). Se forcar_ansatz=True, psi(y*) = y*(1-y*) * Rede(y*)."""
    raw_out = self.rede(y_star)
    if self.forcar_ansatz:
      return y_star * (1.0 - y_star) * raw_out
    return raw_out


# =============================================================================
# 4. SOLVER DE AUTOVALORES E ESTADOS EXCITADOS
# =============================================================================


class EigenSolverSchrodingerPINN:
  """Motor de treinamento para descoberta sequencial de múltiplos estados quânticos (Ground State e Excitados)."""

  def __init__(
      self,
      residual_engine: ResidualSchrodinger,
      lr: float = 1e-3,
      lambda_pde: float = 1.0,
      lambda_bc: float = 50.0,
      lambda_norm: float = 20.0,
      lambda_ortho: float = 40.0,
  ):
    self.engine = residual_engine
    self.lr = lr
    self.lambda_pde = lambda_pde
    self.lambda_bc = lambda_bc
    self.lambda_norm = lambda_norm
    self.lambda_ortho = lambda_ortho

  def resolver_estado(
      self,
      V_star: torch.Tensor,
      y_star: torch.Tensor,
      n_estado: int = 1,
      estados_anteriores: Optional[List[torch.Tensor]] = None,
      max_epocas: int = 1500,
      tol_loss: float = 1e-6,
  ) -> Tuple[SchrodingerPINN, float, float]:
    """Treina uma PINN para convergir ao n-ésimo autovalor e autofunção sob potencial arbitrário V*(y*)."""
    estados_ant = estados_anteriores or []

    # Estimativa inicial de autovalor para poço quântico: E_n ≈ n² * pi² * alpha
    E_init_guess = (n_estado**2) * (np.pi**2) * self.engine.alpha + float(
        torch.mean(V_star).item()
    )

    modelo = SchrodingerPINN(
        hidden_dim=64,
        num_layers=3,
        E_star_inicial=E_init_guess,
        forcar_contorno_ansatz=False,
    ).to(self.engine.device, dtype=torch.float64)

    otimizador = optim.Adam(modelo.parameters(), lr=self.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        otimizador, mode="min", factor=0.5, patience=150
    )

    for epoca in range(1, max_epocas + 1):
      otimizador.zero_grad()

      psi = modelo(y_star)

      # 1. Resíduo de Schrödinger (PDE)
      res = self.engine.residual(psi, modelo.E_star, V_star, y_star)
      loss_pde = torch.mean(res**2)

      # 2. Condições de Contorno de Dirichlet
      loss_bc = self.engine.perda_contorno(psi)

      # 3. Integral de Normalização ∫ |psi|² dy* = 1
      loss_norm = self.engine.perda_normalizacao(psi, y_star)

      # 4. Ortogonalidade com estados de menor energia
      loss_ortho = self.engine.perda_ortogonalidade(
          psi, estados_ant, y_star
      )

      # Perda Total Composta
      loss_total = (
          self.lambda_pde * loss_pde
          + self.lambda_bc * loss_bc
          + self.lambda_norm * loss_norm
          + self.lambda_ortho * loss_ortho
      )

      loss_total.backward()
      otimizador.step()
      scheduler.step(loss_total)

      if loss_total.item() < tol_loss and epoca > 400:
        break

    # Autovalor final em unidades físicas (eV)
    E_star_final = modelo.E_star.item()
    E_eV_final = E_star_final * self.engine.VT_val

    return modelo, E_star_final, E_eV_final


# =============================================================================
# 5. POISSON-SCHRÖDINGER AUTO-CONSISTENTE 1D
# =============================================================================


class SchrodingerPoissonAutoConsistente:
  """Loop auto-consistente completo: V(x) -> Schrödinger PINN -> psi_n, E_n -> n_3D(x) -> Poisson -> V(x)."""

  def __init__(
      self,
      residual_engine: ResidualSchrodinger,
      N_sites: int = 60,
      max_iter_scf: int = 15,
      alpha_damping: float = 0.20,
      E_Fermi_eV: float = 0.10,
  ):
    self.engine = residual_engine
    self.N = N_sites
    self.max_iter = max_iter_scf
    self.alpha = alpha_damping
    self.E_F = E_Fermi_eV
    self.solver = EigenSolverSchrodingerPINN(residual_engine)

    # Discretização espacial
    self.y_star = torch.linspace(
        0.0,
        1.0,
        self.N,
        dtype=torch.float64,
        device=self.engine.device,
        requires_grad=True,
    ).unsqueeze(-1)
    self.x_m = self.y_star.detach() * self.engine.t
    self.dx = self.engine.t / (self.N - 1)

    # Matriz Laplaciana de Poisson 1D convencional: d²phi/dx² = -rho / eps_si
    diag_p = -2.0 / (self.dx**2) * torch.ones(self.N, dtype=torch.float64)
    off_p = 1.0 / (self.dx**2) * torch.ones(self.N - 1, dtype=torch.float64)
    self.M_poisson = (
        torch.diag(diag_p) + torch.diag(off_p, 1) + torch.diag(off_p, -1)
    ).to(self.engine.device)

  def resolver_poisson(
      self, n_3d: torch.Tensor, N_dop_3d: torch.Tensor, V_gate: float
  ) -> torch.Tensor:
    """Resolve a distribuição de potencial eletrostático phi(x) em Volts."""
    rho = self.engine.q * (N_dop_3d - n_3d)
    b = -(rho / self.engine.mat.eps_si)

    A = self.M_poisson.clone()
    # Condições de Contorno: Porta nos dois lados (Double-Gate phi(0) = phi(L) = V_gate)
    A[0, :] = 0.0
    A[0, 0] = 1.0
    b[0] = V_gate

    A[-1, :] = 0.0
    A[-1, -1] = 1.0
    b[-1] = V_gate

    return torch.linalg.solve(A, b)

  def calcular_densidade_eletronica(
      self, autofuncoes: List[torch.Tensor], autovalores_eV: List[float]
  ) -> torch.Tensor:
    """Calcula a densidade volumétrica n_3D(x) [m⁻³] acumulando a ocupação 2D de todas as sub-bandas."""
    n_3d = torch.zeros(
        self.N, dtype=torch.float64, device=self.engine.device
    )
    kBT = self.engine.q * self.engine.VT_val

    for psi_n, E_n in zip(autofuncoes, autovalores_eV):
      # Ocupação 2D da sub-banda: N_2D = (m_dos* * k_B * T / (pi * hbar²)) * ln(1 + exp((E_F - E_n)/kBT))
      coef_dos = (
          (self.engine.mat.m_dos_2d * self.engine.mat.m0 * kBT)
          / (np.pi * (self.engine.hbar**2))
      )
      arg_fermi = max(min((self.E_F - E_n) * self.engine.q / kBT, 80.0), -80.0)
      N_2D = coef_dos * np.log(1.0 + np.exp(arg_fermi))  # m⁻²

      # Distribuição espacial: n_3D(x) = (N_2D / t) * |psi(x/t)|²
      psi_sq = (psi_n.detach().squeeze() ** 2)  # Normalizado em y*
      n_3d += (N_2D / self.engine.t) * psi_sq

    return n_3d

  def executar(
      self,
      V_gate: float,
      N_dop_3d: torch.Tensor,
      num_subbandas: int = 2,
  ) -> Dict[str, object]:
    """Executa o loop auto-consistente completo até convergir o perfil de potencial."""
    phi = torch.full(
        (self.N,),
        V_gate,
        dtype=torch.float64,
        device=self.engine.device,
    )
    n_3d = torch.zeros_like(N_dop_3d)

    for it in range(self.max_iter):
      # Potencial de Confinamento Total: V(x) = V_eletrostatico(x) = -q * phi(x) em Joules
      V_eV = -phi
      V_star = (V_eV / self.engine.VT_val).unsqueeze(-1)

      autofuncoes = []
      autovalores_eV = []

      # 1. Resolução dos estados quânticos via PINN
      for n in range(1, num_subbandas + 1):
        modelo, _, E_eV = self.solver.resolver_estado(
            V_star=V_star,
            y_star=self.y_star,
            n_estado=n,
            estados_anteriores=autofuncoes,
            max_epocas=600,
        )
        with torch.no_grad():
          psi_n = modelo(self.y_star)
        autofuncoes.append(psi_n)
        autovalores_eV.append(E_eV)

      # 2. Carga Quântica n_3D(x)
      n_3d_novo = self.calcular_densidade_eletronica(autofuncoes, autovalores_eV)

      # 3. Poisson Eletrostático
      phi_novo = self.resolver_poisson(n_3d_novo, N_dop_3d, V_gate)

      # 4. Critério de Convergência e Amortecimento Picard
      diff = torch.max(torch.abs(phi_novo - phi)).item()
      phi = (1.0 - self.alpha) * phi + self.alpha * phi_novo
      n_3d = (1.0 - self.alpha) * n_3d + self.alpha * n_3d_novo

      if diff < 1e-4:
        break

    return {
        "phi": phi,
        "V_eV": -phi,
        "n_3d": n_3d,
        "autofuncoes": autofuncoes,
        "autovalores_eV": autovalores_eV,
        "iter": it + 1,
        "y_star": self.y_star.detach().cpu().numpy().squeeze(),
        "x_nm": (self.x_m.detach().cpu().numpy().squeeze() * 1e9),
    }


# =============================================================================
# 6. VALIDAÇÃO ANALÍTICA E DEMONSTRAÇÃO EXECUTÁVEL
# =============================================================================

if __name__ == "__main__":
  torch.manual_seed(42)
  np.random.seed(42)
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 80)
  print(" SCHRÖDINGER 1D PINN SOLVER: VALIDAÇÃO ANALÍTICA & POISSON AUTO-CONSISTENTE")
  print("=" * 80)

  espessura_nm = 2.0  # Nanofio / Nanosheet de 2 nm
  mat_si = ParametrosSilicio(T=300.0)
  residual_engine = ResidualSchrodinger(
      mat=mat_si, t_nm=espessura_nm, T=300.0, device=dispositivo
  )

  print(
      f"Espessura do Canal:               t = {espessura_nm:.1f} nm"
  )
  print(
      f"Massa Efetiva de Confinamento:    m* = {mat_si.m_eff_conf:.2f} m0"
  )
  print(
      f"Energia de Confinamento Base:     E_conf = {residual_engine.E_conf_eV:.4f}"
      " eV"
  )
  print(
      f"Potencial Térmico:                V_T = {residual_engine.VT_val * 1e3:.2f}"
      " mV"
  )
  print(
      f"Fator Adimensional alpha:         alpha = {residual_engine.alpha:.4f}"
  )

  # ---------------------------------------------------------------------------
  # PARTE 1: Validação Analítica Contra o Poço Quântico Infinito (V(x) = 0)
  # ---------------------------------------------------------------------------
  print("\n>>> [1/2] Benchmark Contra Solução Analítica Exata (Poço Infinito)...")
  N_pontos = 100
  y_grid = torch.linspace(
      0.0,
      1.0,
      N_pontos,
      dtype=torch.float64,
      device=dispositivo,
      requires_grad=True,
  ).unsqueeze(-1)
  V_zero = torch.zeros_like(y_grid)

  solver_pinn = EigenSolverSchrodingerPINN(
      residual_engine, lr=2e-3, lambda_norm=30.0, lambda_bc=50.0
  )

  estados_pinn = []
  energias_pinn = []
  energias_exatas = []

  for n in range(1, 4):
    # Solução Exata: E_n = n² * pi² * E_conf / q = n² * pi² * hbar² / (2 m* t²)
    E_exato_eV = (n**2) * (np.pi**2) * residual_engine.E_conf_eV
    energias_exatas.append(E_exato_eV)

    modelo_n, E_star_calc, E_eV_calc = solver_pinn.resolver_estado(
        V_star=V_zero,
        y_star=y_grid,
        n_estado=n,
        estados_anteriores=estados_pinn,
        max_epocas=1200,
    )

    with torch.no_grad():
      psi_calc = modelo_n(y_grid)

    estados_pinn.append(psi_calc)
    energias_pinn.append(E_eV_calc)

    erro_rel = abs(E_eV_calc - E_exato_eV) / E_exato_eV * 100.0
    print(
        f"  Estado n={n} | E_PINN = {E_eV_calc:.5f} eV | E_Exato ="
        f" {E_exato_eV:.5f} eV | Erro Relativo = {erro_rel:.3f}%"
    )

  # ---------------------------------------------------------------------------
  # PARTE 2: Acoplamento Auto-Consistente Schrödinger-Poisson
  # ---------------------------------------------------------------------------
  print("\n>>> [2/2] Executando Loop Auto-Consistente Schrödinger-Poisson...")
  scf_engine = SchrodingerPoissonAutoConsistente(
      residual_engine, N_sites=50, max_iter_scf=6, E_Fermi_eV=0.25
  )

  # Dopagem intrínseca no canal
  N_dop_canal = torch.full(
      (50,), 1e20, dtype=torch.float64, device=dispositivo
  )
  res_scf = scf_engine.executar(
      V_gate=0.30, N_dop_3d=N_dop_canal, num_subbandas=2
  )

  print(
      f"  SCF Convergido em {res_scf['iter']} iterações."
  )
  print(
      f"  Sub-banda Fundamental E1: {res_scf['autovalores_eV'][0]:.4f} eV"
  )
  print(
      f"  Primeira Sub-banda Excitada E2: {res_scf['autovalores_eV'][1]:.4f} eV"
  )
  print(
      f"  Concentração Máxima no Canal:  {torch.max(res_scf['n_3d']).item():.3e}"
      " m⁻³"
  )

  # ---------------------------------------------------------------------------
  # PARTE 3: Painel de Resultados Gráficos
  # ---------------------------------------------------------------------------
  y_np = y_grid.detach().cpu().numpy().squeeze()
  x_nm_val = y_np * espessura_nm

  fig, axs = plt.subplots(1, 2, figsize=(14, 5))

  # 1. Comparação de Autofunções Normalizadas (PINN vs Analítico)
  for n in range(1, 4):
    psi_analitico = np.sqrt(2.0) * np.sin(n * np.pi * y_np)
    psi_pinn_arr = estados_pinn[n - 1].detach().cpu().numpy().squeeze()

    # Ajuste de sinal/fase global para sobreposição
    if np.corrcoef(psi_analitico, psi_pinn_arr)[0, 1] < 0:
      psi_pinn_arr = -psi_pinn_arr

    axs[0].plot(
        x_nm_val,
        psi_analitico,
        "k-",
        alpha=0.6,
        linewidth=1.8,
        label=f"Analítico n={n}" if n == 1 else None,
    )
    axs[0].plot(
        x_nm_val,
        psi_pinn_arr,
        "--",
        linewidth=2.0,
        label=f"PINN $\psi_{n}(x)$ (E={energias_pinn[n-1]:.3f} eV)",
    )

  axs[0].set_title(
      f"Autofunções $\psi_n(x)$ no Poço Quântico ($t={espessura_nm}$ nm)"
  )
  axs[0].set_xlabel("Posição $x$ (nm)")
  axs[0].set_ylabel(r"Amplitude Normalizada $\psi(x)$")
  axs[0].grid(True, linestyle="--", alpha=0.6)
  axs[0].legend(loc="upper right")

  # 2. Distribuição Auto-Consistente de Carga e Potencial
  ax2 = axs[1].twinx()
  axs[1].plot(
      res_scf["x_nm"],
      res_scf["V_eV"].cpu().numpy(),
      "r-",
      linewidth=2.2,
      label=r"Potencial Eletrostático $V(x)$",
  )
  ax2.plot(
      res_scf["x_nm"],
      res_scf["n_3d"].cpu().numpy() * 1e-24,
      "b--",
      linewidth=2.0,
      label=r"Densidade $n_{3D}(x)$ ($10^{18}\text{ cm}^{-3}$)",
  )

  axs[1].set_title(
      f"Schrödinger-Poisson Auto-Consistente ($V_{{gate}} = 0.30$ V)"
  )
  axs[1].set_xlabel("Posição $x$ (nm)")
  axs[1].set_ylabel("Energia Potencial $V(x)$ (eV)", color="r")
  ax2.set_ylabel(r"Densidade Quântica $n_{3D}$ ($10^{24}\text{ m}^{-3}$)", color="b")
  axs[1].grid(True, linestyle="--", alpha=0.6)

  plt.tight_layout()
  plt.show()
