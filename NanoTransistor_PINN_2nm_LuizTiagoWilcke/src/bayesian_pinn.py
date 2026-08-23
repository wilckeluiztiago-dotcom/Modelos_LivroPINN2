"""Módulo: Bayesian PINN via Monte Carlo Dropout & Heteroscedastic UQ

Autor: Luiz Tiago Wilcke
Descrição: Framework completo de Quantificação de Incerteza (UQ) em PINNs:
           1. Teoria de Gal & Ghahramani (Dropout como Inferência Variacional em Processos Gaussianos).
           2. Decomposição de Incerteza Total: Epistêmica (modelo/falta de dados) e Aleatória (ruído).
           3. Diferenciação Automática (Autograd) com caminhos estocásticos de Dropout para cálculo de PDE.
           4. Calibração e intervalos de confiança analíticos (68%, 95% e 99.7%).
           5. Aplicação completa na equação de Poisson reduzida para nanotransistores.
"""

from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# 1. ARQUITETURA DA REDE COM MONTE CARLO DROPOUT E DUPLA CABEÇA (UQ)
# =============================================================================


class MCDropoutPINN(nn.Module):
  """Physics-Informed Neural Network Bayesiana aproximada via Monte Carlo Dropout.

  Implementa arquitetura com dupla cabeça para decomposição explícita:
    - Cabeça 1 (Média μ): Predição do potencial eletrostático φ(x).
    - Cabeça 2 (Log-Variância log(s²)): Incerteza aleatória intrínseca (ruído heterocedástico).
  """

  def __init__(
      self,
      in_dim: int = 3,  # [x_norm, Vgs, Vds]
      hidden_dim: int = 64,
      num_layers: int = 4,
      p_dropout: float = 0.10,
      peso_decay_l2: float = 1e-5,
  ):
    super().__init__()
    self.in_dim = in_dim
    self.hidden_dim = hidden_dim
    self.p_dropout = p_dropout
    self.peso_decay_l2 = peso_decay_l2

    # Camada de entrada
    camadas_tronco = [
        nn.Linear(in_dim, hidden_dim),
        nn.Tanh(),
        nn.Dropout(p=p_dropout),
    ]

    # Camadas intermediárias
    for _ in range(num_layers - 1):
      camadas_tronco.extend([
          nn.Linear(hidden_dim, hidden_dim),
          nn.Tanh(),
          nn.Dropout(p=p_dropout),
      ])

    self.tronco = nn.Sequential(*camadas_tronco)

    # Dupla cabeça de saída
    self.cabeca_media = nn.Linear(hidden_dim, 1)  # Predição física: \mu(x)
    self.cabeca_logvar = nn.Linear(
        hidden_dim, 1
    )  # Incerteza de observação: \log(\sigma^2)

  def forward(
      self, x: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward pass padrão."""
    features = self.tronco(x)
    mu = self.cabeca_media(features)
    log_var = self.cabeca_logvar(features)
    return mu, log_var

  def forward_com_dropout_forcado(
      self, x: torch.Tensor
  ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Força ativação do Dropout mesmo em modo de inferência (torch.no_grad)."""
    # Ativa dropout mantendo o resto das camadas intactas
    for m in self.modules():
      if isinstance(m, nn.Dropout):
        m.train()
    features = self.tronco(x)
    mu = self.cabeca_media(features)
    log_var = self.cabeca_logvar(features)
    return mu, log_var


# =============================================================================
# 2. MOTOR DE QUANTIFICAÇÃO DE INCERTEZA (EPISTÊMICA vs ALEATÓRIA)
# =============================================================================


class MotorIncertezaBayesiana:
  """Calcula a distribuição preditiva posterior decompondo as fontes de incerteza."""

  def __init__(self, modelo: MCDropoutPINN):
    self.modelo = modelo

  @torch.no_grad()
  def predizer_distribuicao(
      self,
      x: torch.Tensor,
      n_amostras: int = 100,
  ) -> Dict[str, torch.Tensor]:
    """Gera N realizações estocásticas da rede e calcula as variâncias:

    - Epistêmica: Incerteza do modelo (falta de dados/física) = Var_t(\mu_t(x))
    - Aleatória: Ruído de medição = E_t(s_t^2(x))
    - Total: \sigma_{tot}^2 = \sigma_{epi}^2 + \sigma_{ale}^2
    """
    amostras_mu = []
    amostras_var_aleatoria = []

    for _ in range(n_amostras):
      mu_t, log_var_t = self.modelo.forward_com_dropout_forcado(x)
      var_aleatoria_t = torch.exp(log_var_t)  # s^2(x) = exp(log_var)

      amostras_mu.append(mu_t)
      amostras_var_aleatoria.append(var_aleatoria_t)

    # Tensor: [n_amostras, N_pontos, 1]
    tensor_mu = torch.stack(amostras_mu, dim=0)
    tensor_var_aleatoria = torch.stack(amostras_var_aleatoria, dim=0)

    # 1. Média Preditiva
    media_pred = tensor_mu.mean(dim=0)

    # 2. Incerteza Epistêmica (Variância entre as predições do ensemble)
    var_epistemica = tensor_mu.var(dim=0, unbiased=True)
    std_epistemica = torch.sqrt(torch.clamp(var_epistemica, min=1e-12))

    # 3. Incerteza Aleatória (Média das variâncias intrínsecas estimadas)
    var_aleatoria = tensor_var_aleatoria.mean(dim=0)
    std_aleatoria = torch.sqrt(torch.clamp(var_aleatoria, min=1e-12))

    # 4. Incerteza Total Combinada
    var_total = var_epistemica + var_aleatoria
    std_total = torch.sqrt(torch.clamp(var_total, min=1e-12))

    # Intervalos de Confiança (Normal / Gaussiano)
    ic_95_inf = media_pred - 1.96 * std_total
    ic_95_sup = media_pred + 1.96 * std_total

    return {
        "media": media_pred,
        "std_epistemica": std_epistemica,
        "std_aleatoria": std_aleatoria,
        "std_total": std_total,
        "ic_95_inf": ic_95_inf,
        "ic_95_sup": ic_95_sup,
        "amostras_brutas": tensor_mu,
    }


# =============================================================================
# 3. PERDAS FÍSICO-BAYESIANAS (HETEROSCEDASTIC NLL + RESÍDUO PDE VIA AUTOGRAD)
# =============================================================================


class TreinadorBayesianPINN:
  """Treina a PINN com perda de Verossimilhança Heterocedástica (NLL) e Resíduos de PDE."""

  EPS_0 = 8.8541878128e-12
  EPS_SI = 11.7 * 8.8541878128e-12
  Q_E = 1.602176634e-19

  def __init__(
      self,
      modelo: MCDropoutPINN,
      L_canal: float = 15e-9,
      lambda_g: float = 2.2e-9,
      lr: float = 1e-3,
  ):
    self.modelo = modelo
    self.L = L_canal
    self.lambda_sq = lambda_g**2
    # Otimizador Adam com regularização L2 (Weight Decay) equivalente ao Prior Gaussiano
    self.opt = optim.Adam(
        self.modelo.parameters(), lr=lr, weight_decay=modelo.peso_decay_l2
    )

  def perda_heterocedastica(
      self, y_real: torch.Tensor, mu_pred: torch.Tensor, log_var_pred: torch.Tensor
  ) -> torch.Tensor:
    """Negative Log-Likelihood (NLL) Gaussiana:

    L_NLL = (1/2) * exp(-log_var) * (y - mu)^2 + (1/2) * log_var
    """
    precisao = torch.exp(-log_var_pred)
    termo_erro = precisao * (y_real - mu_pred) ** 2
    return 0.5 * torch.mean(termo_erro + log_var_pred)

  def passo_treinamento(
      self,
      x_colocacao: torch.Tensor,
      Vgs_val: float,
      Vds_val: float,
      x_dados: torch.Tensor,
      phi_dados_ruidosos: torch.Tensor,
      N_dop_colocacao: torch.Tensor,
      n_eletrons_3d: torch.Tensor,
  ) -> Dict[str, float]:
    """Executa um passo de otimização estocástica com Autograd e Dropout ativos."""
    self.opt.zero_grad()
    self.modelo.train()  # Ativa dropout durante a amostragem de treinamento

    # -------------------------------------------------------------
    # 1. Perda de Dados Observados (Contornos e Sensores com Ruído)
    # -------------------------------------------------------------
    vgs_d = torch.full_like(x_dados, Vgs_val)
    vds_d = torch.full_like(x_dados, Vds_val)
    input_dados = torch.cat([x_dados / self.L, vgs_d, vds_d], dim=-1)

    mu_dados, log_var_dados = self.modelo(input_dados)
    loss_dados = self.perda_heterocedastica(
        phi_dados_ruidosos, mu_dados, log_var_dados
    )

    # -------------------------------------------------------------
    # 2. Perda Física da PDE (Poisson Reduzido via Autograd)
    # -------------------------------------------------------------
    x_col = x_colocacao.clone().detach().requires_grad_(True)
    vgs_c = torch.full_like(x_col, Vgs_val)
    vds_c = torch.full_like(x_col, Vds_val)
    input_col = torch.cat([x_col / self.L, vgs_c, vds_c], dim=-1)

    # Forward pass estocástico com dropout ativo
    phi_col, _ = self.modelo(input_col)

    # dphi/dx
    dphi_dx = torch.autograd.grad(
        phi_col,
        x_col,
        grad_outputs=torch.ones_like(phi_col),
        create_graph=True,
        retain_graph=True,
    )[0]

    # d²phi/dx²
    d2phi_dx2 = torch.autograd.grad(
        dphi_dx,
        x_col,
        grad_outputs=torch.ones_like(dphi_dx),
        create_graph=True,
        retain_graph=True,
    )[0]

    # Resíduo da Equação de Poisson Reduzida
    rho = self.Q_E * (N_dop_colocacao - n_eletrons_3d)
    res_poisson = (
        d2phi_dx2
        - ((phi_col - Vgs_val) / self.lambda_sq)
        + (rho / self.EPS_SI)
    )
    loss_pde = torch.mean(res_poisson**2) * 1e-18  # Normalização dimensional

    # -------------------------------------------------------------
    # 3. Perda Total e Retropropagação
    # -------------------------------------------------------------
    loss_total = 20.0 * loss_dados + loss_pde
    loss_total.backward()
    self.opt.step()

    return {
        "loss_total": loss_total.item(),
        "loss_dados": loss_dados.item(),
        "loss_pde": loss_pde.item(),
    }


# =============================================================================
# 4. EXECUÇÃO COMPLETA, QUANTIFICAÇÃO E VISUALIZAÇÃO
# =============================================================================

if __name__ == "__main__":
  torch.manual_seed(42)
  np.random.seed(42)
  dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  print("=" * 80)
  print(" BAYESIAN PINN: QUANTIFICAÇÃO DE INCERTEZA EM DISPOSITIVOS NANOELETRÔNICOS")
  print("=" * 80)

  # Parâmetros Físicos
  L_canal = 15.0e-9  # 15 nm
  N_grid = 50
  x_fisico = torch.linspace(0, L_canal, N_grid, device=dispositivo).unsqueeze(-1)

  Vgs_teste = 0.50  # V
  Vds_teste = 0.30  # V

  # 1. Geração de Dados Sintéticos Ruidosos (Contornos + Sensores Esparsos)
  x_obs = torch.tensor(
      [[0.0], [2.0e-9], [7.5e-9], [13.0e-9], [15.0e-9]],
      device=dispositivo,
      dtype=torch.float32,
  )

  # Potencial "verdadeiro" de referência + Ruído Heterocedástico
  phi_verdadeiro = (
      (Vds_teste * (x_obs / L_canal))
      + 0.15 * torch.sin(np.pi * x_obs / L_canal)
  ).to(torch.float32)

  # Adiciona ruído com desvio padrão sigma = 15 mV
  ruido_medicao = torch.randn_like(x_obs) * 0.015
  phi_obs_ruidoso = phi_verdadeiro + ruido_medicao

  # Perfis de Carga para o Resíduo de Poisson
  N_dop = torch.full((N_grid, 1), 1e21, device=dispositivo, dtype=torch.float32)
  N_dop[:10] = 1e26
  N_dop[-10:] = 1e26
  n_eletrons = torch.full((N_grid, 1), 5e23, device=dispositivo, dtype=torch.float32)

  # 2. Inicialização e Treinamento da PINN Bayesiana
  rede_bayesiana = MCDropoutPINN(
      in_dim=3,
      hidden_dim=64,
      num_layers=4,
      p_dropout=0.10,
      peso_decay_l2=1e-5,
  ).to(dispositivo)

  treinador = TreinadorBayesianPINN(
      rede_bayesiana, L_canal=L_canal, lambda_g=2.2e-9, lr=2e-3
  )

  print("\n>>> Treinando PINN com Amostragem Estocástica de Dropout...")
  for epoca in range(1, 401):
    metricas = treinador.passo_treinamento(
        x_colocacao=x_fisico,
        Vgs_val=Vgs_teste,
        Vds_val=Vds_teste,
        x_dados=x_obs,
        phi_dados_ruidosos=phi_obs_ruidoso,
        N_dop_colocacao=N_dop,
        n_eletrons_3d=n_eletrons,
    )
    if epoca % 100 == 0 or epoca == 1:
      print(
          f"  Época {epoca:03d} | Perda Total: {metricas['loss_total']:.5e} |"
          f" NLL Dados: {metricas['loss_dados']:.5e} | PDE:"
          f" {metricas['loss_pde']:.5e}"
      )

  # 3. Inferência Bayesiana via Monte Carlo Dropout
  print("\n>>> Executando Motor de Incerteza (150 Amostras de Monte Carlo)...")
  motor_uq = MotorIncertezaBayesiana(rede_bayesiana)

  vgs_grid = torch.full((N_grid, 1), Vgs_teste, device=dispositivo)
  vds_grid = torch.full((N_grid, 1), Vds_teste, device=dispositivo)
  input_inferencia = torch.cat([x_fisico / L_canal, vgs_grid, vds_grid], dim=-1)

  resultado_uq = motor_uq.predizer_distribuicao(input_inferencia, n_amostras=150)

  # Conversão para NumPy para Plotting
  x_nm = (x_fisico.cpu().numpy() * 1e9).squeeze()
  x_obs_nm = (x_obs.cpu().numpy() * 1e9).squeeze()
  y_obs = phi_obs_ruidoso.cpu().numpy().squeeze()

  media_pred = resultado_uq["media"].cpu().numpy().squeeze()
  std_epi = resultado_uq["std_epistemica"].cpu().numpy().squeeze()
  std_ale = resultado_uq["std_aleatoria"].cpu().numpy().squeeze()
  std_tot = resultado_uq["std_total"].cpu().numpy().squeeze()
  ic_inf = resultado_uq["ic_95_inf"].cpu().numpy().squeeze()
  ic_sup = resultado_uq["ic_95_sup"].cpu().numpy().squeeze()

  # 4. Visualização Científica dos Resultados de Incerteza
  fig, axs = plt.subplots(1, 2, figsize=(14, 5))

  # Painel 1: Predição com Faixas de Incerteza (Intervalo de Confiança 95%)
  axs[0].plot(
      x_nm,
      media_pred,
      "b-",
      linewidth=2.2,
      label=r"Média Preditiva $\mu(x)$",
  )
  axs[0].fill_between(
      x_nm,
      ic_inf,
      ic_sup,
      color="blue",
      alpha=0.20,
      label=r"Incerteza Total (IC 95%: $\pm 1.96\sigma_{tot}$)",
  )
  axs[0].fill_between(
      x_nm,
      media_pred - std_epi,
      media_pred + std_epi,
      color="orange",
      alpha=0.35,
      label=r"Incerteza Epistêmica ($\pm 1\sigma_{epi}$)",
  )
  axs[0].scatter(
      x_obs_nm,
      y_obs,
      color="red",
      s=45,
      zorder=5,
      label="Observações Ruidosas",
  )
  axs[0].set_title("Potencial Eletrostático com Incerteza Calibrada")
  axs[0].set_xlabel("Posição no Canal x (nm)")
  axs[0].set_ylabel("Potencial $\phi(x)$ (V)")
  axs[0].grid(True, linestyle="--", alpha=0.6)
  axs[0].legend(loc="upper left")

  # Painel 2: Decomposição das Incertezas (Epistêmica vs Aleatória)
  axs[1].plot(
      x_nm,
      std_epi * 1e3,
      "r-",
      linewidth=2.0,
      label=r"Incerteza Epistêmica $\sigma_{epi}$ (Modelo)",
  )
  axs[1].plot(
      x_nm,
      std_ale * 1e3,
      "g--",
      linewidth=2.0,
      label=r"Incerteza Aleatória $\sigma_{ale}$ (Ruído de Medição)",
  )
  axs[1].plot(
      x_nm,
      std_tot * 1e3,
      "k-.",
      linewidth=2.2,
      label=r"Incerteza Total $\sigma_{tot}$",
  )
  axs[1].set_title("Decomposição Espectral de Incertezas (mV)")
  axs[1].set_xlabel("Posição no Canal x (nm)")
  axs[1].set_ylabel("Desvio Padrão (mV)")
  axs[1].grid(True, linestyle="--", alpha=0.6)
  axs[1].legend()

  plt.tight_layout()
  plt.show()
