"""Módulo: NEGF 1D com Espalhamento Inelástico por Fónons (SCBA)

Autor: Luiz Tiago Wilcke
Descrição: Solver Quântico Dissipativo baseado na Aproximação de Born Auto-Consistente:
           - Fónons Acústicos (Elastic dephasing)
           - Fónons Ópticos (Inelastic energy relaxation com deslocamento E ± ℏω₀)
           - Loop SCBA de ponto fixo com amortecimento Picard
           - Verificação de conservação de corrente local I(x, x+1)
           - Transição contínua entre Transporte Balístico e Difusivo
"""

import math
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

# =============================================================================
# 1. CONSTANTES FÍSICAS E PARÂMETROS DE ESPALHAMENTO
# =============================================================================


class ConstantesTransporte:
  """Constantes fundamentais e fatores de conversão."""

  Q_E: float = 1.602176634e-19  # C
  H_PLANCK: float = 6.62607015e-34  # J·s
  H_BAR: float = 1.054571817e-34  # J·s
  M_E: float = 9.1093837e-31  # kg
  K_B_EV: float = 8.617333262e-5  # eV/K
  G0: float = 2.0 * (1.602176634e-19**2) / 6.62607015e-34  # ~77.48 µS


class ParametrosFonons:
  """Parâmetros físicos de acoplamento elétron-fónon (Silício)."""

  def __init__(
      self,
      hbar_omega_op_eV: float = 0.063,  # Energia do fónon óptico (63 meV no Si)
      D_ac_eV2: float = 1.5e-3,  # Força de espalhamento acústico (eV²)
      D_op_eV2: float = 3.0e-3,  # Força de espalhamento óptico (eV²)
      T: float = 300.0,
  ):
    self.hbar_omega_0 = hbar_omega_op_eV
    self.D_ac = D_ac_eV2
    self.D_op = D_op_eV2
    self.T = T
    self.kBT = max(ConstantesTransporte.K_B_EV * T, 1e-6)

    # Número de ocupação de Bose-Einstein: N0 = 1 / (exp(hbar*w0 / kBT) - 1)
    arg_bose = self.hbar_omega_0 / self.kBT
    self.N0 = 1.0 / (math.exp(arg_bose) - 1.0) if arg_bose < 80.0 else 0.0


# =============================================================================
# 2. SOLVER NEGF-SCBA DISCRETIZADO
# =============================================================================


class PhononSCBA1D:
  """Solver NEGF 1D com Espalhamento Elétron-Fónon via SCBA."""

  def __init__(
      self,
      N_sites: int = 35,
      dx: float = 0.5e-9,  # 0.5 nm
      m_eff: float = 0.25,
      E_F: float = 0.0,
      fonons: Optional[ParametrosFonons] = None,
      device: torch.device = torch.device("cpu"),
  ):
    self.N = N_sites
    self.dx = dx
    self.m_eff = m_eff
    self.E_F = E_F
    self.fonons = fonons or ParametrosFonons()
    self.device = device
    self.kBT = self.fonons.kBT

    # Parâmetro de hopping cinético t0 em eV
    t0_joules = (ConstantesTransporte.H_BAR**2) / (
        2.0 * (self.m_eff * ConstantesTransporte.M_E) * (self.dx**2)
    )
    self.t0 = float(t0_joules / ConstantesTransporte.Q_E)

    # Matriz Hamiltoniana Cinética H0
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
    """Autoenergia analítica para contatos 1D semi-infinitos."""
    theta = (E_grid - U_lead - 2.0 * self.t0) / (2.0 * self.t0)
    sigma = torch.zeros_like(E_grid, dtype=torch.complex128)

    mask_band = torch.abs(theta) <= 1.0
    if mask_band.any():
      th_b = theta[mask_band]
      sigma[mask_band] = torch.complex(
          self.t0 * th_b, -self.t0 * torch.sqrt(1.0 - th_b**2)
      )

    mask_bel = theta < -1.0
    if mask_bel.any():
      th_bel = theta[mask_bel]
      sigma[mask_bel] = torch.complex(
          self.t0 * (th_bel + torch.sqrt(th_bel**2 - 1.0)),
          torch.zeros_like(th_bel),
      )

    mask_abv = theta > 1.0
    if mask_abv.any():
      th_abv = theta[mask_abv]
      sigma[mask_abv] = torch.complex(
          self.t0 * (th_abv - torch.sqrt(th_abv**2 - 1.0)),
          torch.zeros_like(th_abv),
      )

    return sigma

  def _deslocar_energia(self, tensor_E: torch.Tensor, k_desloc: int) -> torch.Tensor:
    """Desloca o tensor no eixo da energia por k_desloc índices para espalhamento óptico."""
    res = torch.zeros_like(tensor_E)
    if k_desloc > 0:
      # E + hbar*omega_0 (índices maiores)
      res[:-k_desloc] = tensor_E[k_desloc:]
    elif k_desloc < 0:
      # E - hbar*omega_0 (índices menores)
      k_abs = abs(k_desloc)
      res[k_abs:] = tensor_E[:-k_abs]
    else:
      res = tensor_E.clone()
    return res

  def resolver_scba(
      self,
      U_potencial: torch.Tensor,
      Vds: float,
      E_grid: torch.Tensor,
      max_iter_scba: int = 40,
      tol_scba: float = 1e-3,
      alpha_scba: float = 0.25,
  ) -> Dict[str, torch.Tensor]:
    """Executa o loop auto-consistente de Born (SCBA) para convergir Sigma_ph e G^<."""
    n_E = E_grid.shape[0]
    dE = (E_grid[1] - E_grid[0]).item()

    # Número de passos de grade correspondentes à energia do fónon óptico
    k_shift = max(1, int(round(self.fonons.hbar_omega_0 / dE)))

    H_total = self.H_kin + torch.diag(U_potencial.to(torch.float64))
    H_c = H_total.to(torch.complex128)

    mu_S = self.E_F
    mu_D = self.E_F - Vds

    f_S = self.fermi_dirac(E_grid, mu_S)
    f_D = self.fermi_dirac(E_grid, mu_D)

    # Autoenergias dos contatos
    sigma_S = self.calcular_sigma_lead(
        E_grid, float(U_potencial[0].item())
    )
    sigma_D = self.calcular_sigma_lead(
        E_grid, float(U_potencial[-1].item())
    )
    gamma_S = -2.0 * sigma_S.imag
    gamma_D = -2.0 * sigma_D.imag

    # Inicialização das autoenergias de espalhamento [n_E, N] (diagonais no espaço real)
    sigma_ph_R_diag = torch.zeros(
        (n_E, self.N), dtype=torch.complex128, device=self.device
    )
    sigma_ph_lesser_diag = torch.zeros(
        (n_E, self.N), dtype=torch.float64, device=self.device
    )

    # Tensores para resolução em lote
    I_exp = self.I_mat.unsqueeze(0).expand(n_E, self.N, self.N)
    A_base = (
        (E_grid.to(torch.complex128) + 1e-7j).view(-1, 1, 1)
        * self.I_mat.unsqueeze(0)
    ) - H_c.unsqueeze(0)
    A_base = A_base.clone()
    A_base[:, 0, 0] -= sigma_S
    A_base[:, -1, -1] -= sigma_D

    for it_scba in range(max_iter_scba):
      # 1. Montagem do sistema linear com autoenergia de fónons
      A_batch = A_base.clone()
      A_batch.diagonal(dim1=-2, dim2=-1)[:] -= sigma_ph_R_diag

      # 2. Resolução da Função de Green Retardada G^R(E)
      G_R = torch.linalg.solve(A_batch, I_exp)
      G_A = G_R.conj().transpose(-2, -1)

      # 3. Função Espectral Local: A_{ii}(E) = i [G^R - G^A]_{ii} = -2 Im[G^R_{ii}]
      A_diag = -2.0 * G_R.diagonal(dim1=-2, dim2=-1).imag  # [n_E, N]

      # 4. In-scattering total: Sigma^in = Gamma_S*f_S + Gamma_D*f_D + Sigma_ph^<
      # G^< = G^R @ Sigma^in @ G^A
      Sigma_in_mat = torch.zeros(
          (n_E, self.N, self.N), dtype=torch.complex128, device=self.device
      )
      Sigma_in_mat[:, 0, 0] = gamma_S * f_S
      Sigma_in_mat[:, -1, -1] = gamma_D * f_D
      Sigma_in_mat.diagonal(dim1=-2, dim2=-1)[:] += sigma_ph_lesser_diag.to(
          torch.complex128
      )

      G_lesser = G_R @ Sigma_in_mat @ G_A
      G_lesser_diag = G_lesser.diagonal(dim1=-2, dim2=-1).real  # [n_E, N]

      # 5. Out-scattering G^> = G^< - i*A
      G_greater_diag = G_lesser_diag + A_diag

      # 6. Atualização das Autoenergias de Espalhamento via SCBA
      # --- Fónons Acústicos (Quase-elástico) ---
      sigma_ac_lesser = self.fonons.D_ac * G_lesser_diag
      gamma_ac = self.fonons.D_ac * A_diag

      # --- Fónons Ópticos (Inelástico com deslocamento E ± ℏω₀) ---
      G_lesser_emissao = self._deslocar_energia(
          G_lesser_diag, +k_shift
      )  # E + hbar*w0
      G_lesser_absorcao = self._deslocar_energia(
          G_lesser_diag, -k_shift
      )  # E - hbar*w0

      A_emissao = self._deslocar_energia(A_diag, +k_shift)
      A_absorcao = self._deslocar_energia(A_diag, -k_shift)

      sigma_op_lesser = self.fonons.D_op * (
          self.fonons.N0 * G_lesser_absorcao
          + (self.fonons.N0 + 1.0) * G_lesser_emissao
      )
      gamma_op = self.fonons.D_op * (
          self.fonons.N0 * A_absorcao + (self.fonons.N0 + 1.0) * A_emissao
      )

      # Autoenergias totais propostas
      sigma_ph_lesser_novo = sigma_ac_lesser + sigma_op_lesser
      gamma_ph_novo = gamma_ac + gamma_op
      sigma_ph_R_novo = torch.complex(
          torch.zeros_like(gamma_ph_novo), -0.5 * gamma_ph_novo
      )

      # 7. Verificação de Convergência do Ponto Fixo SCBA
      erro_scba = torch.max(
          torch.abs(sigma_ph_lesser_novo - sigma_ph_lesser_diag)
      ).item() / (torch.max(torch.abs(sigma_ph_lesser_diag)).item() + 1e-9)

      # Mistura Picard
      sigma_ph_lesser_diag = (
          1.0 - alpha_scba
      ) * sigma_ph_lesser_diag + alpha_scba * sigma_ph_lesser_novo
      sigma_ph_R_diag = (
          1.0 - alpha_scba
      ) * sigma_ph_R_diag + alpha_scba * sigma_ph_R_novo

      if erro_scba < tol_scba:
        break

    # 8. Cálculo das Grandezas Físicas Finais
    # Densidade Linear: n_1D(x) = (2 / (2*pi * dx)) * int G^<_{ii}(E) dE
    integrando_n = (2.0 / (2.0 * np.pi * self.dx)) * G_lesser_diag
    n_1d = torch.sum(integrando_n, dim=0) * dE

    # Corrente local entre sítios adjacentes I(x, x+1):
    # I_{i, i+1} = (4q² / h) * int dE Re[ -t0 * G^<_{i+1, i}(E) ]
    G_lesser_offdiag = G_lesser[
        :, 1:, :-1
    ]  # Elementos sub-diagonais [n_E, N-1]
    integrando_corrente_local = (
        ConstantesTransporte.G0
        * 2.0
        * (-self.t0 * G_lesser_offdiag.real.diagonal(dim1=-2, dim2=-1))
    )
    corrente_perfil = (
        torch.sum(integrando_corrente_local, dim=0) * dE
    )  # [N-1]

    # Espectro de Transmissão Efetivo
    G_R_0N = G_R[:, 0, -1]
    T_efetivo = torch.clamp(
        gamma_S * gamma_D * (G_R_0N.real**2 + G_R_0N.imag**2), min=0.0
    )

    return {
        "n_1d": n_1d,
        "corrente_perfil": corrente_perfil,
        "I_terminal_media": float(torch.mean(corrente_perfil).item()),
        "T_efetivo": T_efetivo,
        "ldos": (A_diag / (2.0 * np.pi * self.dx)),
        "G_lesser": G_lesser,
        "iter_scba": it_scba + 1,
        "convergido": erro_scba < tol_scba,
    }


# =============================================================================
# 3. DEMONSTRAÇÃO E TRANSIÇÃO BALÍSTICO-DIFUSIVA
# =============================================================================

if __name__ == "__main__":
  torch.manual_seed(42)
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 80)
  print(
      " SIMULAÇÃO NEGF-SCBA: TRANSIÇÃO CONTÍNUA BALÍSTICO-DIFUSIVA POR FÓNONS"
  )
  print("=" * 80)

  N_grid = 35
  dx_nm = 0.5  # 0.5 nm -> Canal de 17 nm
  L_canal_nm = (N_grid - 1) * dx_nm
  x_nm = np.linspace(0, L_canal_nm, N_grid)

  # Grid de Energia com resolução para capturar o fónon óptico (63 meV)
  E_grid = torch.linspace(-0.3, 1.2, 360, dtype=torch.float64, device=dispositivo)

  # Perfil de Barreira Eletrostática no Canal (Vgs ativado, Vds aplicado)
  Vds_bias = 0.35  # V
  U_pot = 0.25 * torch.sin(
      torch.linspace(0, np.pi, N_grid, dtype=torch.float64, device=dispositivo)
  )
  U_pot += torch.linspace(
      0.0, -Vds_bias, N_grid, dtype=torch.float64, device=dispositivo
  )

  # Configuração de Três Regimes de Transporte
  regimes = [
      {
          "nome": "Balístico Puro (Sem Fónons)",
          "D_ac": 0.0,
          "D_op": 0.0,
          "cor": "blue",
      },
      {
          "nome": "Acoplamento Fraco (Quasi-Balístico)",
          "D_ac": 1.5e-3,
          "D_op": 2.0e-3,
          "cor": "green",
      },
      {
          "nome": "Acoplamento Forte (Difusivo / Dissipativo)",
          "D_ac": 6.0e-3,
          "D_op": 9.0e-3,
          "cor": "red",
      },
  ]

  resultados = []

  for reg in regimes:
    fonons = ParametrosFonons(
        hbar_omega_op_eV=0.063,
        D_ac_eV2=reg["D_ac"],
        D_op_eV2=reg["D_op"],
        T=300.0,
    )
    solver = PhononSCBA1D(
        N_sites=N_grid,
        dx=dx_nm * 1e-9,
        m_eff=0.25,
        E_F=0.05,
        fonons=fonons,
        device=dispositivo,
    )

    print(f"\n>>> Executando Regime: {reg['nome']}...")
    res = solver.resolver_scba(
        U_pot, Vds=Vds_bias, E_grid=E_grid, max_iter_scba=35, tol_scba=1e-3
    )

    I_media_uA = res["I_terminal_media"] * 1e6
    var_corrente = torch.std(res["corrente_perfil"]).item() * 1e6
    print(
        f"    SCBA Convergido: {res['convergido']} em {res['iter_scba']}"
        " iterações"
    )
    print(
        f"    Corrente Média I_DS: {I_media_uA:8.4f} µA | Desvio ao longo do"
        f" canal: {var_corrente:.3e} µA"
    )

    resultados.append((reg, res))

  # Visualização da Degradação da Corrente e Relaxação Espectral
  fig, axs = plt.subplots(1, 3, figsize=(16, 4.8))

  # 1. Perfil Espacial de Corrente Local I(x, x+1) - Conservação
  for reg, res in resultados:
    x_mid_nm = (x_nm[:-1] + x_nm[1:]) / 2.0
    axs[0].plot(
        x_mid_nm,
        res["corrente_perfil"].cpu().numpy() * 1e6,
        linewidth=2.0,
        color=reg["cor"],
        label=reg["nome"],
    )
  axs[0].set_title("Conservação da Corrente Local $I(x, x+1)$")
  axs[0].set_xlabel("Posição no Canal $x$ (nm)")
  axs[0].set_ylabel("Corrente Local $I_{DS}$ (µA)")
  axs[0].grid(True, linestyle="--", alpha=0.6)
  axs[0].legend(loc="lower left", fontsize=8.5)

  # 2. Espectro de Transmissão Efetiva T(E)
  for reg, res in resultados:
    axs[1].plot(
        E_grid.cpu().numpy(),
        res["T_efetivo"].cpu().numpy(),
        linewidth=2.0,
        color=reg["cor"],
        label=reg["nome"],
    )
  axs[1].set_title("Espectro de Transmissão $T(E)$ sob Espalhamento")
  axs[1].set_xlabel("Energia $E$ (eV)")
  axs[1].set_ylabel("Transmissão $T(E)$")
  axs[1].grid(True, linestyle="--", alpha=0.6)
  axs[1].legend(loc="upper left", fontsize=8.5)

  # 3. Densidade Linear de Elétrons n_1D(x)
  for reg, res in resultados:
    axs[2].plot(
        x_nm,
        res["n_1d"].cpu().numpy() * 1e-8,
        linewidth=2.0,
        color=reg["cor"],
        label=reg["nome"],
    )
  axs[2].set_title("Acúmulo de Carga $n_{1D}(x)$")
  axs[2].set_xlabel("Posição no Canal $x$ (nm)")
  axs[2].set_ylabel(r"Densidade Linear ($10^8$ m⁻¹)")
  axs[2].grid(True, linestyle="--", alpha=0.6)
  axs[2].legend(loc="upper right", fontsize=8.5)

  plt.tight_layout()
  plt.show()
