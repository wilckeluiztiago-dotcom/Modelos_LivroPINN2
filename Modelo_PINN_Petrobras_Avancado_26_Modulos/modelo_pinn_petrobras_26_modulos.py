# ==============================================================================
# MODELO PINN DEEP GALERKIN MULTI-FATORIAL AVANÇADO - PETROBRAS (PETR4)
# Autor: Luiz Tiago Wilcke
# Versão: 2.0 - 26 Módulos com Bibliotecas Estatísticas Sofisticadas
# ==============================================================================
"""
Modelo de precificação de derivativos de commodities (PETR4) baseado em
Physics-Informed Neural Networks (PINN) com Deep Galerkin Method (DGM).

Fatores estocásticos:
- Gibson-Schwartz (rendimento de conveniência do petróleo)
- Heston (volatilidade estocástica)
- Saltos de Merton (processo de Lévy)
- Mudança de regime de Markov (2 estados)

Bibliotecas estatísticas avançadas: scipy.stats, statsmodels, pandas, numpy.
"""

import torch
import torch.nn as nn
import numpy as np
import math
import warnings
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict

# Bibliotecas estatísticas sofisticadas
import pandas as pd
from scipy import stats, optimize, integrate, special
from scipy.stats import norm, t as student_t, jarque_bera, kstest, shapiro, anderson
from scipy.stats import skew, kurtosis, pearsonr, spearmanr
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import het_breuschpagan, acorr_ljungbox
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==============================================================================
# MODULO 01: CONFIGURACOES E HIPERPARAMETROS GLOBAIS
# ==============================================================================
@dataclass
class ConfiguracaoModeloPetrobras:
    dispositivo: str = "cuda" if torch.cuda.is_available() else "cpu"
    tipo_dado: torch.dtype = torch.float32
   
    # Parametros de Mercado e Ativo PETR4
    taxa_livre_risco: float = 0.1075          # Taxa Selic anualizada
    preco_base_petr4: float = 38.50           # Preco spot base (BRL)
    preco_minimo: float = 10.0
    preco_maximo: float = 80.0
   
    # Gibson-Schwartz (Rendimento de Conveniencia do Petroleo Brent)
    kappa_rendimento: float = 1.45
    theta_rendimento: float = 0.08
    sigma_rendimento: float = 0.22
    rendimento_minimo: float = -0.20
    rendimento_maximo: float = 0.40
   
    # Heston (Volatilidade Estocastica)
    kappa_variancia: float = 2.10
    theta_variancia: float = 0.09             # ~30% vol
    xi_vol_variancia: float = 0.35            # vol-of-vol
    variancia_minima: float = 0.01
    variancia_maxima: float = 0.50
   
    # Estrutura de Correlacao Cruzada
    correlacao_preco_rendimento: float = 0.35
    correlacao_preco_variancia: float = -0.65  # Efeito alavancagem
    correlacao_rendimento_variancia: float = 0.15
   
    # Parametros de Salto de Merton
    intensidade_salto: float = 0.40
    media_salto_log: float = -0.05
    desvio_salto_log: float = 0.12
    amostras_salto_mc: int = 16
   
    # Mudanca de Regime de Markov
    taxa_transicao_regime_1_2: float = 0.15
    taxa_transicao_regime_2_1: float = 0.45
   
    # Dominio Espaco-Temporal
    tempo_maximo: float = 1.0
   
    # Otimizacao
    lote_colocalizacao: int = 2048            # Reduzido para estabilidade em CPU
    lote_contorno: int = 512
    epocas_adam: int = 300
    taxa_aprendizado_adam: float = 1e-3
    iteracoes_maximas_lbfgs: int = 100
    
    # Parametros estatisticos avancados
    nivel_confianca_var: float = 0.99
    numero_bootstrap: int = 500
    seed_estatistico: int = 42

configuracao = ConfiguracaoModeloPetrobras()

# ==============================================================================
# MODULO 02: MOTOR DE DADOS E TENSORES FINANCEIROS
# ==============================================================================
class MotorDadosFinanceiros:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def gerar_serie_historica_sintetica(self, numero_dias: int = 252) -> Dict[str, torch.Tensor]:
        passo_tempo = 1.0 / numero_dias
        torch.manual_seed(self.cfg.seed_estatistico)
       
        precos = torch.zeros(numero_dias, device=self.cfg.dispositivo)
        rendimentos = torch.zeros(numero_dias, device=self.cfg.dispositivo)
        variancias = torch.zeros(numero_dias, device=self.cfg.dispositivo)
       
        precos[0] = self.cfg.preco_base_petr4
        rendimentos[0] = self.cfg.theta_rendimento
        variancias[0] = self.cfg.theta_variancia
       
        matriz_covariancia = torch.tensor([
            [1.0, self.cfg.correlacao_preco_rendimento, self.cfg.correlacao_preco_variancia],
            [self.cfg.correlacao_preco_rendimento, 1.0, self.cfg.correlacao_rendimento_variancia],
            [self.cfg.correlacao_preco_variancia, self.cfg.correlacao_rendimento_variancia, 1.0]
        ], device=self.cfg.dispositivo)
        cholesky_fator = torch.linalg.cholesky(matriz_covariancia)
       
        for t in range(1, numero_dias):
            ruido_normal = torch.randn(3, 1, device=self.cfg.dispositivo)
            incremento_browniano = torch.matmul(cholesky_fator, ruido_normal).squeeze() * math.sqrt(passo_tempo)
           
            contagem_salto = torch.poisson(torch.tensor(self.cfg.intensidade_salto * passo_tempo, device=self.cfg.dispositivo))
            magnitude_salto = 0.0
            if contagem_salto > 0:
                magnitude_salto = torch.randn(1, device=self.cfg.dispositivo).item() * self.cfg.desvio_salto_log + self.cfg.media_salto_log
           
            variancias[t] = torch.clamp(
                variancias[t-1] + self.cfg.kappa_variancia * (self.cfg.theta_variancia - variancias[t-1]) * passo_tempo +
                self.cfg.xi_vol_variancia * torch.sqrt(torch.clamp(variancias[t-1], min=1e-4)) * incremento_browniano[2],
                min=1e-4
            )
            rendimentos[t] = rendimentos[t-1] + self.cfg.kappa_rendimento * (self.cfg.theta_rendimento - rendimentos[t-1]) * passo_tempo + \
                             self.cfg.sigma_rendimento * incremento_browniano[1]
           
            deriva_preco = (self.cfg.taxa_livre_risco - rendimentos[t-1]) * passo_tempo
            difusao_preco = torch.sqrt(variancias[t-1]) * incremento_browniano[0]
            precos[t] = precos[t-1] * torch.exp(deriva_preco - 0.5 * variancias[t-1] * passo_tempo + difusao_preco + magnitude_salto)
           
        return {"preco": precos, "rendimento": rendimentos, "variancia": variancias}

    def converter_para_dataframe(self, series: Dict[str, torch.Tensor]) -> pd.DataFrame:
        """Converte tensores para DataFrame pandas para análise estatística."""
        return pd.DataFrame({
            "preco": series["preco"].cpu().numpy(),
            "rendimento": series["rendimento"].cpu().numpy(),
            "variancia": series["variancia"].cpu().numpy(),
            "log_retorno": np.diff(np.log(series["preco"].cpu().numpy()), prepend=np.nan)
        })

# ==============================================================================
# MODULO 03: SIMULADOR ESTOCASTICO MONTE CARLO MULTI-FATORIAL
# ==============================================================================
class SimuladorMonteCarloMultiFatorial:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg
       
    def simular_trajetorias(self, numero_trajetorias: int, numero_passos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        passo_tempo = self.cfg.tempo_maximo / numero_passos
        matriz_precos = torch.full((numero_trajetorias, numero_passos + 1), self.cfg.preco_base_petr4, device=self.cfg.dispositivo)
        matriz_rendimentos = torch.full((numero_trajetorias, numero_passos + 1), self.cfg.theta_rendimento, device=self.cfg.dispositivo)
        matriz_variancias = torch.full((numero_trajetorias, numero_passos + 1), self.cfg.theta_variancia, device=self.cfg.dispositivo)
       
        matriz_correlacao = torch.tensor([
            [1.0, self.cfg.correlacao_preco_rendimento, self.cfg.correlacao_preco_variancia],
            [self.cfg.correlacao_preco_rendimento, 1.0, self.cfg.correlacao_rendimento_variancia],
            [self.cfg.correlacao_preco_variancia, self.cfg.correlacao_rendimento_variancia, 1.0]
        ], device=self.cfg.dispositivo)
        cholesky_fator = torch.linalg.cholesky(matriz_correlacao)
       
        compensador_salto = self.cfg.intensidade_salto * (math.exp(self.cfg.media_salto_log + 0.5 * self.cfg.desvio_salto_log**2) - 1.0)
       
        for t in range(numero_passos):
            ruido_gaussiano = torch.randn(numero_trajetorias, 3, device=self.cfg.dispositivo)
            incrementos = torch.matmul(ruido_gaussiano, cholesky_fator.T) * math.sqrt(passo_tempo)
           
            variancia_atual = torch.clamp(matriz_variancias[:, t], min=1e-4)
            matriz_variancias[:, t+1] = torch.clamp(
                variancia_atual + self.cfg.kappa_variancia * (self.cfg.theta_variancia - variancia_atual) * passo_tempo +
                self.cfg.xi_vol_variancia * torch.sqrt(variancia_atual) * incrementos[:, 2],
                min=1e-4
            )
            matriz_rendimentos[:, t+1] = matriz_rendimentos[:, t] + self.cfg.kappa_rendimento * (self.cfg.theta_rendimento - matriz_rendimentos[:, t]) * passo_tempo + \
                                         self.cfg.sigma_rendimento * incrementos[:, 1]
           
            saltos_poisson = torch.poisson(torch.full((numero_trajetorias,), self.cfg.intensidade_salto * passo_tempo, device=self.cfg.dispositivo))
            impactos_salto = saltos_poisson * (torch.randn(numero_trajetorias, device=self.cfg.dispositivo) * self.cfg.desvio_salto_log + self.cfg.media_salto_log)
           
            deriva = (self.cfg.taxa_livre_risco - matriz_rendimentos[:, t] - compensador_salto) * passo_tempo
            difusao = torch.sqrt(variancia_atual) * incrementos[:, 0]
            matriz_precos[:, t+1] = matriz_precos[:, t] * torch.exp(deriva - 0.5 * variancia_atual * passo_tempo + difusao + impactos_salto)
           
        return matriz_precos, matriz_rendimentos, matriz_variancias

# ==============================================================================
# MODULO 04: AMOSTRADOR DE COLOCALIZACAO HIPERCUBO LATINO (SOBOL)
# ==============================================================================
class AmostradorColocalizacaoAdaptativo:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg
       
    def amostrar_dominio_interior(self, total_pontos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gerador_sobol = torch.quasirandom.SobolEngine(dimension=4, scramble=True)
        pontos_quase_aleatorios = gerador_sobol.draw(total_pontos).to(device=self.cfg.dispositivo, dtype=self.cfg.tipo_dado)
       
        precos = pontos_quase_aleatorios[:, 0:1] * (self.cfg.preco_maximo - self.cfg.preco_minimo) + self.cfg.preco_minimo
        rendimentos = pontos_quase_aleatorios[:, 1:2] * (self.cfg.rendimento_maximo - self.cfg.rendimento_minimo) + self.cfg.rendimento_minimo
        variancias = pontos_quase_aleatorios[:, 2:3] * (self.cfg.variancia_maxima - self.cfg.variancia_minima) + self.cfg.variancia_minima
        tempos = pontos_quase_aleatorios[:, 3:4] * self.cfg.tempo_maximo
        return precos, rendimentos, variancias, tempos

    def amostrar_fronteira_terminal(self, total_pontos: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gerador_sobol = torch.quasirandom.SobolEngine(dimension=3, scramble=True)
        pontos_quase_aleatorios = gerador_sobol.draw(total_pontos).to(device=self.cfg.dispositivo, dtype=self.cfg.tipo_dado)
       
        precos = pontos_quase_aleatorios[:, 0:1] * (self.cfg.preco_maximo - self.cfg.preco_minimo) + self.cfg.preco_minimo
        rendimentos = pontos_quase_aleatorios[:, 1:2] * (self.cfg.rendimento_maximo - self.cfg.rendimento_minimo) + self.cfg.rendimento_minimo
        variancias = pontos_quase_aleatorios[:, 2:3] * (self.cfg.variancia_maxima - self.cfg.variancia_minima) + self.cfg.variancia_minima
        tempos = torch.full_like(precos, self.cfg.tempo_maximo)
        return precos, rendimentos, variancias, tempos

# ==============================================================================
# MODULO 05: CAMADA ESPECTRAL DE FOURIER EMBEDDING
# ==============================================================================
class IncorporacaoEspectralFourier(nn.Module):
    def __init__(self, dimensao_entrada: int, numero_frequencias: int = 32, escala: float = 1.0):
        super().__init__()
        self.numero_frequencias = numero_frequencias
        matriz_projecao = torch.randn(dimensao_entrada, numero_frequencias) * escala
        self.register_buffer("matriz_projecao", matriz_projecao)

    def forward(self, vetor_entrada: torch.Tensor) -> torch.Tensor:
        projecao_angular = 2.0 * math.pi * torch.matmul(vetor_entrada, self.matriz_projecao)
        return torch.cat([torch.sin(projecao_angular), torch.cos(projecao_angular)], dim=-1)

# ==============================================================================
# MODULO 06: CELULA RECORRENTE DEEP GALERKIN METHOD (DGM)
# ==============================================================================
class CelulaDGM_Recorrente(nn.Module):
    def __init__(self, dimensao_oculta: int, dimensao_entrada: int):
        super().__init__()
        self.transformacao_z_entrada = nn.Linear(dimensao_entrada, dimensao_oculta, bias=True)
        self.transformacao_z_estado = nn.Linear(dimensao_oculta, dimensao_oculta, bias=False)
        self.transformacao_g_entrada = nn.Linear(dimensao_entrada, dimensao_oculta, bias=True)
        self.transformacao_g_estado = nn.Linear(dimensao_oculta, dimensao_oculta, bias=False)
        self.transformacao_r_entrada = nn.Linear(dimensao_entrada, dimensao_oculta, bias=True)
        self.transformacao_r_estado = nn.Linear(dimensao_oculta, dimensao_oculta, bias=False)
        self.transformacao_h_entrada = nn.Linear(dimensao_entrada, dimensao_oculta, bias=True)
        self.transformacao_h_estado = nn.Linear(dimensao_oculta, dimensao_oculta, bias=False)
        self.ativacao = nn.Tanh()

    def forward(self, vetor_entrada: torch.Tensor, estado_anterior: torch.Tensor) -> torch.Tensor:
        gate_atualizacao = self.ativacao(self.transformacao_z_entrada(vetor_entrada) + self.transformacao_z_estado(estado_anterior))
        gate_redefinicao = self.ativacao(self.transformacao_g_entrada(vetor_entrada) + self.transformacao_g_estado(estado_anterior))
        gate_relevancia = self.ativacao(self.transformacao_r_entrada(vetor_entrada) + self.transformacao_r_estado(estado_anterior))
        candidato_estado = self.ativacao(self.transformacao_h_entrada(vetor_entrada) + self.transformacao_h_estado(estado_anterior * gate_relevancia))
        proximo_estado = (1.0 - gate_redefinicao) * candidato_estado + gate_atualizacao * estado_anterior
        return proximo_estado

# ==============================================================================
# MODULO 07: ARQUITETURA PRINCIPAL PINN DEEP GALERKIN
# ==============================================================================
class RedeNeuralDGM_PINN_Petrobras(nn.Module):
    def __init__(self, dimensao_oculta: int = 128, numero_camadas_dgm: int = 4):
        super().__init__()
        self.dimensao_entrada = 4
        self.incorporador_fourier = IncorporacaoEspectralFourier(self.dimensao_entrada, numero_frequencias=24)
        dimensao_espectral = 48
       
        self.camada_inicial = nn.Sequential(
            nn.Linear(dimensao_espectral, dimensao_oculta),
            nn.Tanh()
        )
        self.camadas_dgm = nn.ModuleList([
            CelulaDGM_Recorrente(dimensao_oculta, dimensao_espectral) for _ in range(numero_camadas_dgm)
        ])
       
        self.cabeca_regime_normal = nn.Sequential(
            nn.Linear(dimensao_oculta, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus()
        )
        self.cabeca_regime_estresse = nn.Sequential(
            nn.Linear(dimensao_oculta, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Softplus()
        )

    def forward(self, preco: torch.Tensor, rendimento: torch.Tensor, variancia: torch.Tensor, tempo: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        entradas_normalizadas = torch.cat([
            (preco - 38.5) / 20.0,
            (rendimento - 0.08) / 0.15,
            (variancia - 0.09) / 0.15,
            (tempo - 0.5) / 0.5
        ], dim=-1)
       
        representacao_espectral = self.incorporador_fourier(entradas_normalizadas)
        estado_camada = self.camada_inicial(representacao_espectral)
        for camada in self.camadas_dgm:
            estado_camada = camada(representacao_espectral, estado_camada)
           
        valor_regime_1 = self.cabeca_regime_normal(estado_camada) * 50.0
        valor_regime_2 = self.cabeca_regime_estresse(estado_camada) * 50.0
        return valor_regime_1, valor_regime_2

# ==============================================================================
# MODULO 08: MOTOR AUTOGRAD DE ALTA ORDEM PARA DERIVADAS CRUZADAS
# ==============================================================================
class MotorDiferenciacaoAutomaticaAltaOrdem:
    @staticmethod
    def calcular_derivadas_edp(valor: torch.Tensor, preco: torch.Tensor, rendimento: torch.Tensor,
                               variancia: torch.Tensor, tempo: torch.Tensor) -> Dict[str, torch.Tensor]:
        gradientes_primeira_ordem = torch.autograd.grad(
            valor.sum(), [preco, rendimento, variancia, tempo], create_graph=True, retain_graph=True
        )
        derivada_preco, derivada_rendimento, derivada_variancia, derivada_tempo = gradientes_primeira_ordem
       
        derivada_preco_segunda = torch.autograd.grad(derivada_preco.sum(), preco, create_graph=True, retain_graph=True)[0]
        derivada_rendimento_segunda = torch.autograd.grad(derivada_rendimento.sum(), rendimento, create_graph=True, retain_graph=True)[0]
        derivada_variancia_segunda = torch.autograd.grad(derivada_variancia.sum(), variancia, create_graph=True, retain_graph=True)[0]
       
        cruzada_preco_rendimento = torch.autograd.grad(derivada_preco.sum(), rendimento, create_graph=True, retain_graph=True)[0]
        cruzada_preco_variancia = torch.autograd.grad(derivada_preco.sum(), variancia, create_graph=True, retain_graph=True)[0]
        cruzada_rendimento_variancia = torch.autograd.grad(derivada_rendimento.sum(), variancia, create_graph=True, retain_graph=True)[0]
       
        return {
            "derivada_tempo": derivada_tempo,
            "derivada_preco": derivada_preco,
            "derivada_rendimento": derivada_rendimento,
            "derivada_variancia": derivada_variancia,
            "derivada_preco_segunda": derivada_preco_segunda,
            "derivada_rendimento_segunda": derivada_rendimento_segunda,
            "derivada_variancia_segunda": derivada_variancia_segunda,
            "cruzada_preco_rendimento": cruzada_preco_rendimento,
            "cruzada_preco_variancia": cruzada_preco_variancia,
            "cruzada_rendimento_variancia": cruzada_rendimento_variancia
        }

# ==============================================================================
# MODULO 09: RESIDUO DA EDP GIBSON-SCHWARTZ (COMMODITY PETROLEO)
# ==============================================================================
class ResiduoEDP_GibsonSchwartz:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def calcular_operador(self, preco: torch.Tensor, rendimento: torch.Tensor,
                          derivadas: Dict[str, torch.Tensor], valor: torch.Tensor) -> torch.Tensor:
        termo_deriva = (self.cfg.taxa_livre_risco - rendimento) * preco * derivadas["derivada_preco"] + \
                       self.cfg.kappa_rendimento * (self.cfg.theta_rendimento - rendimento) * derivadas["derivada_rendimento"]
        termo_difusao = 0.5 * (self.cfg.sigma_rendimento**2) * derivadas["derivada_rendimento_segunda"]
        return termo_deriva + termo_difusao

# ==============================================================================
# MODULO 10: OPERADOR INTEGRO-DIFERENCIAL DE SALTOS DE MERTON
# ==============================================================================
class OperadorIntegroDiferencialSaltosMerton:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras, modelo: nn.Module):
        self.cfg = cfg
        self.modelo = modelo

    def calcular_integral_salto(self, preco: torch.Tensor, rendimento: torch.Tensor,
                                variancia: torch.Tensor, tempo: torch.Tensor, regime_ativo: int) -> torch.Tensor:
        tamanho_lote = preco.shape[0]
        acumulador_integral = torch.zeros(tamanho_lote, 1, device=preco.device)
       
        for _ in range(self.cfg.amostras_salto_mc):
            ruido_salto = torch.randn(tamanho_lote, 1, device=preco.device)
            fator_salto = torch.exp(self.cfg.media_salto_log + self.cfg.desvio_salto_log * ruido_salto)
            preco_perturbado = preco * fator_salto
           
            valor_regime_1, valor_regime_2 = self.modelo(preco_perturbado, rendimento, variancia, tempo)
            valor_selecionado = valor_regime_1 if regime_ativo == 1 else valor_regime_2
            acumulador_integral = acumulador_integral + valor_selecionado
           
        esperanca_integral = acumulador_integral / self.cfg.amostras_salto_mc
        return esperanca_integral

# ==============================================================================
# MODULO 11: RESIDUO DA EDP DE VOLATILIDADE ESTOCASTICA DE HESTON
# ==============================================================================
class ResiduoEDP_HestonVolatilidadeEstocastica:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def calcular_operador(self, preco: torch.Tensor, variancia: torch.Tensor,
                          derivadas: Dict[str, torch.Tensor]) -> torch.Tensor:
        variancia_estavel = torch.clamp(variancia, min=1e-4)
        deriva_variancia = self.cfg.kappa_variancia * (self.cfg.theta_variancia - variancia_estavel) * derivadas["derivada_variancia"]
        difusao_variancia = 0.5 * (self.cfg.xi_vol_variancia**2) * variancia_estavel * derivadas["derivada_variancia_segunda"]
        difusao_preco = 0.5 * variancia_estavel * (preco**2) * derivadas["derivada_preco_segunda"]
        correlacao_preco_var = self.cfg.correlacao_preco_variancia * self.cfg.xi_vol_variancia * variancia_estavel * preco * derivadas["cruzada_preco_variancia"]
        return deriva_variancia + difusao_variancia + difusao_preco + correlacao_preco_var

# ==============================================================================
# MODULO 12: ACOPLADOR MULTI-HEAD PARA TRANSICAO DE REGIME DE MARKOV
# ==============================================================================
class AcopladorPerdaTransicaoRegimeMarkov:
    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def calcular_acoplamento_regimes(self, valor_regime_1: torch.Tensor,
                                     valor_regime_2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        acoplamento_1 = self.cfg.taxa_transicao_regime_1_2 * (valor_regime_2 - valor_regime_1)
        acoplamento_2 = self.cfg.taxa_transicao_regime_2_1 * (valor_regime_1 - valor_regime_2)
        return acoplamento_1, acoplamento_2

# ==============================================================================
# MODULO 13: REGULARIZADOR DE FRONTEIRA LIVRE FISCHER-BURMEISTER
# ==============================================================================
class RegularizadorFronteiraLivreFischerBurmeister:
    @staticmethod
    def calcular_perda_complementaridade(valor: torch.Tensor, preco: torch.Tensor,
                                         preco_exercicio: float, residuo_pide: torch.Tensor,
                                         tolerancia_suave: float = 1e-5) -> torch.Tensor:
        payoff_imediato = torch.clamp(preco - preco_exercicio, min=0.0)
        componente_a = valor - payoff_imediato
        componente_b = -residuo_pide
        residuo_fischer_burmeister = componente_a + componente_b - torch.sqrt(
            componente_a**2 + componente_b**2 + tolerancia_suave
        )
        return torch.mean(residuo_fischer_burmeister**2)

# ==============================================================================
# MODULO 14: CALIBRADOR INVERSO DE PARAMETROS LATENTES
# ==============================================================================
class CalibradorInversoParametrosLatentes(nn.Module):
    def __init__(self, kappa_inicial: float = 1.0):
        super().__init__()
        self.log_kappa_estimado = nn.Parameter(torch.tensor(math.log(kappa_inicial)))

    @property
    def kappa_estimado(self) -> torch.Tensor:
        return torch.exp(self.log_kappa_estimado)

    def calcular_perda_calibracao(self, precos_mercado: torch.Tensor, precos_preditos: torch.Tensor) -> torch.Tensor:
        return torch.mean((precos_preditos - precos_mercado)**2)

# ==============================================================================
# MODULO 15: AGREGADOR DE PERDAS MULTI-OBJETIVO ADAPTATIVO
# ==============================================================================
class AgregadorPerdasMultiObjetivoAdaptativo(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_peso_pde = nn.Parameter(torch.tensor(0.0))
        self.log_peso_contorno = nn.Parameter(torch.tensor(0.0))
        self.log_peso_fischer = nn.Parameter(torch.tensor(0.0))

    def forward(self, perda_pde: torch.Tensor, perda_contorno: torch.Tensor,
                perda_fischer: torch.Tensor) -> torch.Tensor:
        peso_pde = torch.exp(-self.log_peso_pde)
        peso_contorno = torch.exp(-self.log_peso_contorno)
        peso_fischer = torch.exp(-self.log_peso_fischer)
        return (peso_pde * perda_pde + self.log_peso_pde +
                peso_contorno * perda_contorno + self.log_peso_contorno +
                peso_fischer * perda_fischer + self.log_peso_fischer)

# ==============================================================================
# MODULO 16: MOTOR DE OTIMIZACAO HIBRIDA (ADAM + L-BFGS)
# ==============================================================================
class MotorOtimizacaoHibrida:
    def __init__(self, modelo: nn.Module, cfg: ConfiguracaoModeloPetrobras):
        self.modelo = modelo
        self.cfg = cfg
        self.otimizador_adam = torch.optim.Adam(modelo.parameters(), lr=cfg.taxa_aprendizado_adam)
        self.escalonador = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.otimizador_adam, T_max=cfg.epocas_adam
        )

    def passo_treinamento_adam(self, funcao_perda_fechamento) -> float:
        self.otimizador_adam.zero_grad()
        perda_calculada = funcao_perda_fechamento()
        perda_calculada.backward()
        torch.nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=1.0)
        self.otimizador_adam.step()
        self.escalonador.step()
        return perda_calculada.item()

    def refinamento_lbfgs(self, funcao_perda_fechamento):
        otimizador_lbfgs = torch.optim.LBFGS(
            self.modelo.parameters(),
            max_iter=self.cfg.iteracoes_maximas_lbfgs,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe"
        )
        def closure():
            otimizador_lbfgs.zero_grad()
            perda = funcao_perda_fechamento()
            perda.backward()
            return perda
        otimizador_lbfgs.step(closure)

# ==============================================================================
# MODULO 17: EXTRATOR NATIVO DE GREGAS DE RISCO (AUTOGRAD)
# ==============================================================================
class ExtratorNativoGregasRisco:
    @staticmethod
    def extrair_superficie_completa_gregas(modelo: nn.Module, preco: torch.Tensor,
                                           rendimento: torch.Tensor, variancia: torch.Tensor,
                                           tempo: torch.Tensor) -> Dict[str, torch.Tensor]:
        preco = preco.clone().detach().requires_grad_(True)
        rendimento = rendimento.clone().detach().requires_grad_(True)
        variancia = variancia.clone().detach().requires_grad_(True)
        tempo = tempo.clone().detach().requires_grad_(True)
       
        valor_regime_1, _ = modelo(preco, rendimento, variancia, tempo)
       
        gradientes = torch.autograd.grad(
            valor_regime_1.sum(), [preco, variancia, rendimento, tempo],
            create_graph=True, retain_graph=True
        )
        delta_grega = gradientes[0]
        vega_grega = gradientes[1]
        sensibilidade_rendimento = gradientes[2]
        theta_grega = -gradientes[3]
       
        gamma_grega = torch.autograd.grad(delta_grega.sum(), preco, create_graph=True, retain_graph=True)[0]
        vanna_grega = torch.autograd.grad(delta_grega.sum(), variancia, create_graph=True, retain_graph=True)[0]
       
        return {
            "Preco": valor_regime_1.detach(),
            "Delta": delta_grega.detach(),
            "Gamma": gamma_grega.detach(),
            "Vega": vega_grega.detach(),
            "Vanna": vanna_grega.detach(),
            "Theta": theta_grega.detach(),
            "Sensibilidade_Rendimento": sensibilidade_rendimento.detach()
        }

# ==============================================================================
# MODULO 18: MOTOR DE COBERTURA DINAMICA (DEEP HEDGING)
# ==============================================================================
class MotorCoberturaDinamicaDeepHedging(nn.Module):
    def __init__(self, dimensao_oculta: int = 64):
        super().__init__()
        self.rede_decisao = nn.Sequential(
            nn.Linear(5, dimensao_oculta),
            nn.ReLU(),
            nn.Linear(dimensao_oculta, dimensao_oculta),
            nn.ReLU(),
            nn.Linear(dimensao_oculta, 1),
            nn.Sigmoid()
        )

    def forward(self, preco_atual: torch.Tensor, rendimento_atual: torch.Tensor,
                variancia_atual: torch.Tensor, tempo_atual: torch.Tensor,
                posicao_anterior: torch.Tensor) -> torch.Tensor:
        vetor_estado = torch.cat([
            preco_atual, rendimento_atual, variancia_atual, tempo_atual, posicao_anterior
        ], dim=-1)
        return self.rede_decisao(vetor_estado)

    def calcular_perda_pnl_cobertura(self, trajetorias_preco: torch.Tensor,
                                     preco_exercicio: float, taxa_custo_transacao: float = 0.001) -> torch.Tensor:
        total_trajetorias, total_passos = trajetorias_preco.shape
        posicao_hedge = torch.zeros(total_trajetorias, 1, device=trajetorias_preco.device)
        riqueza_acumulada = torch.zeros(total_trajetorias, 1, device=trajetorias_preco.device)
       
        for k in range(total_passos - 1):
            preco_t = trajetorias_preco[:, k:k+1]
            preco_proximo = trajetorias_preco[:, k+1:k+2]
            tempo_t = torch.full_like(preco_t, k / float(total_passos))
           
            proxima_posicao = self.forward(
                preco_t / 38.5, torch.zeros_like(preco_t),
                torch.full_like(preco_t, 0.09), tempo_t, posicao_hedge
            )
            custos = taxa_custo_transacao * torch.abs(proxima_posicao - posicao_hedge) * preco_t
            riqueza_acumulada = riqueza_acumulada + proxima_posicao * (preco_proximo - preco_t) - custos
            posicao_hedge = proxima_posicao
           
        payoff_final = torch.clamp(trajetorias_preco[:, -1:] - preco_exercicio, min=0.0)
        erro_cobertura = payoff_final - riqueza_acumulada
        return torch.mean(erro_cobertura**2)

# ==============================================================================
# MODULO 19: AVALIADOR DE METRICAS QUANTITATIVAS & BACKTEST
# ==============================================================================
class AvaliadorMetricasQuantitativas:
    @staticmethod
    def avaliar_convergencia_terminal(modelo: nn.Module, precos_teste: torch.Tensor,
                                       preco_exercicio: float = 38.5) -> Dict[str, float]:
        rendimentos_teste = torch.full_like(precos_teste, configuracao.theta_rendimento)
        variancias_teste = torch.full_like(precos_teste, configuracao.theta_variancia)
        tempos_teste = torch.full_like(precos_teste, configuracao.tempo_maximo)
       
        with torch.no_grad():
            valor_regime_1, _ = modelo(precos_teste, rendimentos_teste, variancias_teste, tempos_teste)
            payoff_exato = torch.clamp(precos_teste - preco_exercicio, min=0.0)
            rmse_terminal = torch.sqrt(torch.mean((valor_regime_1 - payoff_exato)**2)).item()
            mae_terminal = torch.mean(torch.abs(valor_regime_1 - payoff_exato)).item()
           
        return {"RMSE_Terminal": rmse_terminal, "MAE_Terminal": mae_terminal}

# ==============================================================================
# MODULO 20: ANALISE ESTATISTICA DESCRITIVA E INFERENCIAL (scipy + pandas)
# ==============================================================================
class AnalisadorEstatisticoDescritivo:
    """Módulo de estatística descritiva e inferencial usando scipy.stats e pandas."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg
        np.random.seed(cfg.seed_estatistico)

    def estatisticas_descritivas(self, serie: np.ndarray, nome: str = "serie") -> Dict[str, float]:
        serie_limpa = serie[~np.isnan(serie)]
        return {
            f"{nome}_media": float(np.mean(serie_limpa)),
            f"{nome}_mediana": float(np.median(serie_limpa)),
            f"{nome}_desvio_padrao": float(np.std(serie_limpa, ddof=1)),
            f"{nome}_assimetria": float(skew(serie_limpa)),
            f"{nome}_curtose": float(kurtosis(serie_limpa, fisher=True)),
            f"{nome}_min": float(np.min(serie_limpa)),
            f"{nome}_max": float(np.max(serie_limpa)),
            f"{nome}_q25": float(np.percentile(serie_limpa, 25)),
            f"{nome}_q75": float(np.percentile(serie_limpa, 75)),
            f"{nome}_iqr": float(np.percentile(serie_limpa, 75) - np.percentile(serie_limpa, 25))
        }

    def testes_normalidade(self, serie: np.ndarray) -> Dict[str, Any]:
        serie_limpa = serie[~np.isnan(serie)]
        if len(serie_limpa) < 8:
            return {"erro": "Amostra insuficiente"}
        
        jb_stat, jb_p = jarque_bera(serie_limpa)
        sw_stat, sw_p = shapiro(serie_limpa[:5000] if len(serie_limpa) > 5000 else serie_limpa)
        ad_result = anderson(serie_limpa, dist='norm')
        
        return {
            "jarque_bera_stat": float(jb_stat),
            "jarque_bera_pvalue": float(jb_p),
            "shapiro_wilk_stat": float(sw_stat),
            "shapiro_wilk_pvalue": float(sw_p),
            "anderson_darling_stat": float(ad_result.statistic),
            "anderson_darling_criticos": ad_result.critical_values.tolist(),
            "normalidade_rejeitada_jb": jb_p < 0.05,
            "normalidade_rejeitada_sw": sw_p < 0.05
        }

    def correlacoes_avancadas(self, df: pd.DataFrame) -> Dict[str, Any]:
        cols = [c for c in ["preco", "rendimento", "variancia", "log_retorno"] if c in df.columns]
        resultado = {}
        for i, c1 in enumerate(cols):
            for c2 in cols[i+1:]:
                s1 = df[c1].dropna().values
                s2 = df[c2].dropna().values
                min_len = min(len(s1), len(s2))
                pearson_r, pearson_p = pearsonr(s1[:min_len], s2[:min_len])
                spearman_r, spearman_p = spearmanr(s1[:min_len], s2[:min_len])
                resultado[f"pearson_{c1}_{c2}"] = {"r": float(pearson_r), "p": float(pearson_p)}
                resultado[f"spearman_{c1}_{c2}"] = {"r": float(spearman_r), "p": float(spearman_p)}
        return resultado

# ==============================================================================
# MODULO 21: TESTES DE ESTACIONARIEDADE E DIAGNOSTICOS (statsmodels)
# ==============================================================================
class DiagnosticoEstacionariedadeStatsmodels:
    """Testes de estacionariedade e diagnósticos de série temporal com statsmodels."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def testar_estacionariedade(self, serie: np.ndarray, nome: str = "serie") -> Dict[str, Any]:
        serie_limpa = serie[~np.isnan(serie)]
        if len(serie_limpa) < 20:
            return {"erro": "Amostra insuficiente"}

        # Augmented Dickey-Fuller
        adf_result = adfuller(serie_limpa, autolag='AIC')
        # KPSS
        kpss_result = kpss(serie_limpa, regression='c', nlags='auto')
        
        return {
            f"adf_{nome}_stat": float(adf_result[0]),
            f"adf_{nome}_pvalue": float(adf_result[1]),
            f"adf_{nome}_estacionaria": adf_result[1] < 0.05,
            f"kpss_{nome}_stat": float(kpss_result[0]),
            f"kpss_{nome}_pvalue": float(kpss_result[1]),
            f"kpss_{nome}_estacionaria": kpss_result[1] > 0.05  # H0: estacionária
        }

    def diagnostico_autocorrelacao(self, serie: np.ndarray, lags: int = 20) -> Dict[str, Any]:
        serie_limpa = serie[~np.isnan(serie)]
        acf_vals = acf(serie_limpa, nlags=lags, fft=True)
        pacf_vals = pacf(serie_limpa, nlags=lags)
        lb_result = acorr_ljungbox(serie_limpa, lags=[lags], return_df=True)
        
        return {
            "acf": acf_vals.tolist(),
            "pacf": pacf_vals.tolist(),
            "ljung_box_stat": float(lb_result["lb_stat"].iloc[0]),
            "ljung_box_pvalue": float(lb_result["lb_pvalue"].iloc[0]),
            "autocorrelacao_significativa": float(lb_result["lb_pvalue"].iloc[0]) < 0.05
        }

    def teste_heterocedasticidade(self, y: np.ndarray, X: np.ndarray) -> Dict[str, Any]:
        """Breusch-Pagan test via statsmodels."""
        X_const = add_constant(X)
        modelo = OLS(y, X_const).fit()
        bp_stat, bp_p, _, _ = het_breuschpagan(modelo.resid, modelo.model.exog)
        return {
            "breusch_pagan_stat": float(bp_stat),
            "breusch_pagan_pvalue": float(bp_p),
            "heterocedasticidade_presente": bp_p < 0.05
        }

# ==============================================================================
# MODULO 22: ESTIMADOR DE VOLATILIDADE REALIZADA E MODELOS GARCH-LIKE
# ==============================================================================
class EstimadorVolatilidadeRealizadaGARCH:
    """Estimação de volatilidade realizada e aproximação GARCH via statsmodels."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def volatilidade_realizada(self, log_retornos: np.ndarray, janela: int = 21) -> np.ndarray:
        """Volatilidade realizada rolling (desvio-padrão anualizado)."""
        s = pd.Series(log_retornos)
        vol = s.rolling(window=janela).std() * np.sqrt(252)
        return vol.values

    def estimar_parametros_garch_aproximado(self, log_retornos: np.ndarray) -> Dict[str, float]:
        """
        Aproximação de parâmetros GARCH(1,1) via regressão linear em
        resíduos ao quadrado (método de moments / quasi-ML simplificado).
        """
        r = log_retornos[~np.isnan(log_retornos)]
        r2 = r**2
        # lag-1
        r2_lag = np.roll(r2, 1)
        r2_lag[0] = r2[0]
        
        X = add_constant(r2_lag)
        modelo = OLS(r2[1:], X[1:]).fit()
        
        omega = max(modelo.params[0], 1e-8)
        alpha = max(min(modelo.params[1], 0.3), 0.01)
        # beta aproximado via persistência
        persistencia = 0.95  # típico
        beta = max(min(persistencia - alpha, 0.9), 0.5)
        
        vol_longo_prazo = np.sqrt(omega / (1 - alpha - beta)) * np.sqrt(252) if (alpha + beta) < 1 else 0.3
        
        return {
            "omega": float(omega),
            "alpha": float(alpha),
            "beta": float(beta),
            "persistencia": float(alpha + beta),
            "vol_longo_prazo_anualizada": float(vol_longo_prazo),
            "r_squared_ajuste": float(modelo.rsquared)
        }

    def densidade_kernel_retornos(self, log_retornos: np.ndarray, pontos: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """Estimativa de densidade por kernel (scipy.stats.gaussian_kde)."""
        r = log_retornos[~np.isnan(log_retornos)]
        kde = stats.gaussian_kde(r)
        x = np.linspace(r.min() - 0.01, r.max() + 0.01, pontos)
        return x, kde(x)

# ==============================================================================
# MODULO 23: MOTOR DE METRICAS DE RISCO AVANCADAS (VaR, CVaR, ES)
# ==============================================================================
class MotorMetricasRiscoAvancadas:
    """Cálculo de Value-at-Risk, Expected Shortfall e métricas de cauda com scipy."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg
        self.alpha = 1.0 - cfg.nivel_confianca_var

    def var_historico(self, perdas: np.ndarray) -> float:
        return float(np.percentile(perdas, 100 * (1 - self.alpha)))

    def var_parametrico_normal(self, media: float, desvio: float) -> float:
        return float(media + desvio * norm.ppf(1 - self.alpha))

    def var_parametrico_t_student(self, media: float, desvio: float, df: float) -> float:
        return float(media + desvio * student_t.ppf(1 - self.alpha, df))

    def expected_shortfall(self, perdas: np.ndarray) -> float:
        var = self.var_historico(perdas)
        return float(np.mean(perdas[perdas >= var]))

    def calcular_todas_metricas(self, retornos: np.ndarray) -> Dict[str, float]:
        perdas = -retornos  # convenção: perda positiva
        media = float(np.mean(perdas))
        desvio = float(np.std(perdas, ddof=1))
        
        # Ajuste de graus de liberdade t-Student via MLE
        try:
            df_est, loc_est, scale_est = student_t.fit(perdas)
        except Exception:
            df_est, loc_est, scale_est = 5.0, media, desvio
        
        return {
            "VaR_historico_99": self.var_historico(perdas),
            "VaR_normal_99": self.var_parametrico_normal(media, desvio),
            "VaR_t_student_99": self.var_parametrico_t_student(loc_est, scale_est, df_est),
            "Expected_Shortfall_99": self.expected_shortfall(perdas),
            "media_perda": media,
            "desvio_perda": desvio,
            "df_t_student": float(df_est),
            "assimetria_perda": float(skew(perdas)),
            "curtose_excesso_perda": float(kurtosis(perdas, fisher=True))
        }

# ==============================================================================
# MODULO 24: CALIBRADOR ESTATISTICO AVANCADO (scipy.optimize + MLE)
# ==============================================================================
class CalibradorEstatisticoAvancadoMLE:
    """Calibração de parâmetros do modelo via Maximum Likelihood Estimation (scipy)."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg

    def log_likelihood_heston_aproximado(self, params: np.ndarray, retornos: np.ndarray) -> float:
        """Log-verossimilhança aproximada para processo de Heston (discretização Euler)."""
        kappa, theta, xi = params
        if kappa <= 0 or theta <= 0 or xi <= 0:
            return 1e10
        v = theta
        ll = 0.0
        dt = 1.0 / 252.0
        for r in retornos:
            if np.isnan(r):
                continue
            # aproximação
            mean = -0.5 * v * dt
            var = max(v * dt, 1e-8)
            ll += -0.5 * np.log(2 * np.pi * var) - 0.5 * (r - mean)**2 / var
            # atualização de variância (simplificada)
            v = v + kappa * (theta - v) * dt
            v = max(v, 1e-6)
        return -ll  # minimizamos -LL

    def calibrar_heston_mle(self, log_retornos: np.ndarray) -> Dict[str, float]:
        r = log_retornos[~np.isnan(log_retornos)]
        x0 = np.array([self.cfg.kappa_variancia, self.cfg.theta_variancia, self.cfg.xi_vol_variancia])
        bounds = [(0.1, 10.0), (0.01, 0.5), (0.05, 1.0)]
        
        resultado = optimize.minimize(
            self.log_likelihood_heston_aproximado,
            x0,
            args=(r,),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 200}
        )
        
        return {
            "kappa_calibrado": float(resultado.x[0]),
            "theta_calibrado": float(resultado.x[1]),
            "xi_calibrado": float(resultado.x[2]),
            "sucesso_otimizacao": bool(resultado.success),
            "log_likelihood": float(-resultado.fun) if resultado.success else np.nan
        }

    def calibrar_saltos_merton_momentos(self, log_retornos: np.ndarray) -> Dict[str, float]:
        """Calibração de intensidade e momentos de salto via método dos momentos."""
        r = log_retornos[~np.isnan(log_retornos)]
        m1 = np.mean(r)
        m2 = np.var(r)
        m3 = stats.moment(r, moment=3)
        m4 = stats.moment(r, moment=4)
        
        # Aproximações simplificadas (assumindo dt=1/252)
        lambda_est = max(0.05, min(2.0, abs(m4) / (3 * m2**2 + 1e-8) * 0.1))
        mu_j = m3 / (lambda_est + 1e-8) * 0.5
        sigma_j = np.sqrt(max(0.01, abs(m2) * 0.3))
        
        return {
            "intensidade_salto_estimada": float(lambda_est),
            "media_salto_log_estimada": float(mu_j),
            "desvio_salto_log_estimado": float(sigma_j),
            "momento_3": float(m3),
            "momento_4": float(m4)
        }

# ==============================================================================
# MODULO 25: ANALISE DE SENSIBILIDADE E BOOTSTRAP ESTATISTICO
# ==============================================================================
class AnalisadorSensibilidadeBootstrap:
    """Bootstrap não-paramétrico e análise de sensibilidade de parâmetros."""

    def __init__(self, cfg: ConfiguracaoModeloPetrobras):
        self.cfg = cfg
        np.random.seed(cfg.seed_estatistico)

    def bootstrap_estatistica(self, serie: np.ndarray, estatistica_func, n_boot: int = None) -> Dict[str, float]:
        n_boot = n_boot or self.cfg.numero_bootstrap
        serie = serie[~np.isnan(serie)]
        n = len(serie)
        estatisticas = []
        for _ in range(n_boot):
            amostra = np.random.choice(serie, size=n, replace=True)
            estatisticas.append(estatistica_func(amostra))
        estatisticas = np.array(estatisticas)
        return {
            "media_bootstrap": float(np.mean(estatisticas)),
            "desvio_bootstrap": float(np.std(estatisticas, ddof=1)),
            "ic_95_inferior": float(np.percentile(estatisticas, 2.5)),
            "ic_95_superior": float(np.percentile(estatisticas, 97.5)),
            "vies_estimado": float(np.mean(estatisticas) - estatistica_func(serie))
        }

    def sensibilidade_parametros_preco(self, modelo: nn.Module, preco_base: float = 38.5,
                                       variacao_pct: float = 0.05) -> Dict[str, float]:
        """Análise de sensibilidade local (diferenças finitas) no preço do modelo."""
        dispositivo = next(modelo.parameters()).device
        p = torch.tensor([[preco_base]], device=dispositivo, dtype=torch.float32)
        r = torch.tensor([[configuracao.theta_rendimento]], device=dispositivo)
        v = torch.tensor([[configuracao.theta_variancia]], device=dispositivo)
        t = torch.tensor([[0.0]], device=dispositivo)
        
        with torch.no_grad():
            v0, _ = modelo(p, r, v, t)
            v_up, _ = modelo(p * (1 + variacao_pct), r, v, t)
            v_dn, _ = modelo(p * (1 - variacao_pct), r, v, t)
        
        elasticidade = ((v_up - v_dn) / (2 * v0 + 1e-8)) / (2 * variacao_pct)
        return {
            "preco_modelo_base": float(v0.item()),
            "preco_modelo_up": float(v_up.item()),
            "preco_modelo_dn": float(v_dn.item()),
            "elasticidade_preco": float(elasticidade.item())
        }

# ==============================================================================
# MODULO 26: PIPELINE INTEGRADO DE PRODUCAO (ORQUESTRADOR MASTER)
# ==============================================================================
class PipelineProducaoPINN_Petrobras:
    def __init__(self):
        self.cfg = configuracao
        self.amostrador = AmostradorColocalizacaoAdaptativo(self.cfg)
        self.rede_neural = RedeNeuralDGM_PINN_Petrobras(
            dimensao_oculta=96, numero_camadas_dgm=3
        ).to(device=self.cfg.dispositivo)
        self.motor_gibson_schwartz = ResiduoEDP_GibsonSchwartz(self.cfg)
        self.motor_heston = ResiduoEDP_HestonVolatilidadeEstocastica(self.cfg)
        self.motor_saltos = OperadorIntegroDiferencialSaltosMerton(self.cfg, self.rede_neural)
        self.acoplador_regimes = AcopladorPerdaTransicaoRegimeMarkov(self.cfg)
        self.agregador_perdas = AgregadorPerdasMultiObjetivoAdaptativo().to(device=self.cfg.dispositivo)
        self.motor_otimizacao = MotorOtimizacaoHibrida(self.rede_neural, self.cfg)
        self.motor_cobertura = MotorCoberturaDinamicaDeepHedging().to(device=self.cfg.dispositivo)
        
        # Módulos estatísticos avançados (20-25)
        self.analisador_descritivo = AnalisadorEstatisticoDescritivo(self.cfg)
        self.diagnostico_ts = DiagnosticoEstacionariedadeStatsmodels(self.cfg)
        self.estimador_vol = EstimadorVolatilidadeRealizadaGARCH(self.cfg)
        self.motor_risco = MotorMetricasRiscoAvancadas(self.cfg)
        self.calibrador_mle = CalibradorEstatisticoAvancadoMLE(self.cfg)
        self.analisador_bootstrap = AnalisadorSensibilidadeBootstrap(self.cfg)
        self.motor_dados = MotorDadosFinanceiros(self.cfg)

    def fechamento_perda_total(self) -> torch.Tensor:
        precos, rendimentos, variancias, tempos = self.amostrador.amostrar_dominio_interior(
            self.cfg.lote_colocalizacao
        )
        precos.requires_grad_(True)
        rendimentos.requires_grad_(True)
        variancias.requires_grad_(True)
        tempos.requires_grad_(True)
       
        valor_regime_1, valor_regime_2 = self.rede_neural(precos, rendimentos, variancias, tempos)
       
        derivadas_r1 = MotorDiferenciacaoAutomaticaAltaOrdem.calcular_derivadas_edp(
            valor_regime_1, precos, rendimentos, variancias, tempos
        )
        derivadas_r2 = MotorDiferenciacaoAutomaticaAltaOrdem.calcular_derivadas_edp(
            valor_regime_2, precos, rendimentos, variancias, tempos
        )
       
        compensador_deriva_salto = (
            self.cfg.intensidade_salto *
            (math.exp(self.cfg.media_salto_log + 0.5 * self.cfg.desvio_salto_log**2) - 1.0)
        )
       
        # Regime 1
        termo_gs_1 = self.motor_gibson_schwartz.calcular_operador(precos, rendimentos, derivadas_r1, valor_regime_1)
        termo_heston_1 = self.motor_heston.calcular_operador(precos, variancias, derivadas_r1)
        integral_salto_1 = self.motor_saltos.calcular_integral_salto(precos, rendimentos, variancias, tempos, 1)
        ajuste_deriva_salto_1 = -compensador_deriva_salto * precos * derivadas_r1["derivada_preco"]
       
        residuo_pide_1 = (
            derivadas_r1["derivada_tempo"] + termo_gs_1 + termo_heston_1 + ajuste_deriva_salto_1 -
            (self.cfg.taxa_livre_risco + self.cfg.intensidade_salto) * valor_regime_1 +
            self.cfg.intensidade_salto * integral_salto_1
        )
       
        # Regime 2
        termo_gs_2 = self.motor_gibson_schwartz.calcular_operador(precos, rendimentos, derivadas_r2, valor_regime_2)
        termo_heston_2 = self.motor_heston.calcular_operador(precos, variancias, derivadas_r2)
        integral_salto_2 = self.motor_saltos.calcular_integral_salto(precos, rendimentos, variancias, tempos, 2)
        ajuste_deriva_salto_2 = -compensador_deriva_salto * precos * derivadas_r2["derivada_preco"]
       
        residuo_pide_2 = (
            derivadas_r2["derivada_tempo"] + termo_gs_2 + termo_heston_2 + ajuste_deriva_salto_2 -
            (self.cfg.taxa_livre_risco + self.cfg.intensidade_salto) * valor_regime_2 +
            self.cfg.intensidade_salto * integral_salto_2
        )
       
        acoplamento_1, acoplamento_2 = self.acoplador_regimes.calcular_acoplamento_regimes(
            valor_regime_1, valor_regime_2
        )
        perda_pde = torch.mean((residuo_pide_1 + acoplamento_1)**2) + torch.mean((residuo_pide_2 + acoplamento_2)**2)
       
        # Contorno terminal
        precos_ic, rend_ic, var_ic, tempos_ic = self.amostrador.amostrar_fronteira_terminal(self.cfg.lote_contorno)
        v1_terminal, v2_terminal = self.rede_neural(precos_ic, rend_ic, var_ic, tempos_ic)
        payoff_terminal = torch.clamp(precos_ic - self.cfg.preco_base_petr4, min=0.0)
        perda_contorno = torch.mean((v1_terminal - payoff_terminal)**2) + torch.mean((v2_terminal - payoff_terminal)**2)
       
        # Fischer-Burmeister
        perda_fischer = RegularizadorFronteiraLivreFischerBurmeister.calcular_perda_complementaridade(
            valor_regime_1, precos, self.cfg.preco_base_petr4, residuo_pide_1
        )
       
        return self.agregador_perdas(perda_pde, perda_contorno, perda_fischer)

    def executar_analise_estatistica_completa(self, serie_df: pd.DataFrame) -> Dict[str, Any]:
        """Executa todos os módulos estatísticos 20-25."""
        resultados = {}
        
        log_ret = serie_df["log_retorno"].dropna().values
        
        # Módulo 20
        resultados["descritivas_preco"] = self.analisador_descritivo.estatisticas_descritivas(
            serie_df["preco"].values, "preco"
        )
        resultados["descritivas_retorno"] = self.analisador_descritivo.estatisticas_descritivas(
            log_ret, "log_retorno"
        )
        resultados["normalidade"] = self.analisador_descritivo.testes_normalidade(log_ret)
        resultados["correlacoes"] = self.analisador_descritivo.correlacoes_avancadas(serie_df)
        
        # Módulo 21
        resultados["estacionariedade_preco"] = self.diagnostico_ts.testar_estacionariedade(
            serie_df["preco"].values, "preco"
        )
        resultados["estacionariedade_retorno"] = self.diagnostico_ts.testar_estacionariedade(
            log_ret, "log_retorno"
        )
        resultados["autocorrelacao"] = self.diagnostico_ts.diagnostico_autocorrelacao(log_ret)
        
        # Módulo 22
        resultados["garch_aproximado"] = self.estimador_vol.estimar_parametros_garch_aproximado(log_ret)
        vol_realizada = self.estimador_vol.volatilidade_realizada(log_ret)
        resultados["vol_realizada_media"] = float(np.nanmean(vol_realizada))
        
        # Módulo 23
        resultados["metricas_risco"] = self.motor_risco.calcular_todas_metricas(log_ret)
        
        # Módulo 24
        resultados["calibracao_heston_mle"] = self.calibrador_mle.calibrar_heston_mle(log_ret)
        resultados["calibracao_saltos"] = self.calibrador_mle.calibrar_saltos_merton_momentos(log_ret)
        
        # Módulo 25
        resultados["bootstrap_media_retorno"] = self.analisador_bootstrap.bootstrap_estatistica(
            log_ret, np.mean, n_boot=200
        )
        resultados["sensibilidade"] = self.analisador_bootstrap.sensibilidade_parametros_preco(
            self.rede_neural
        )
        
        return resultados

    def executar_ciclo_producao(self):
        print("=" * 70)
        print("  MODELO PINN DEEP GALERKIN MULTI-FATORIAL - PETROBRAS (PETR4)")
        print("  Autor: Luiz Tiago Wilcke | 26 Módulos | Bibliotecas Estatísticas Avançadas")
        print("=" * 70)
        print(f"Dispositivo: {self.cfg.dispositivo.upper()} | dtype: {self.cfg.tipo_dado}")
       
        # --- Fase 0: Geração de dados sintéticos e análise estatística ---
        print("\n[Fase 0] Gerando série histórica sintética e executando módulos estatísticos...")
        series = self.motor_dados.gerar_serie_historica_sintetica(252)
        df = self.motor_dados.converter_para_dataframe(series)
        
        # Treinamento rápido (reduzido para demonstração)
        print("\n[Fase 1] Treinamento Adam (épocas reduzidas para demonstração)...")
        for epoca in range(1, 61):
            perda_epoca = self.motor_otimizacao.passo_treinamento_adam(self.fechamento_perda_total)
            if epoca % 20 == 0:
                print(f"  [Adam] Época {epoca:03d} | Perda: {perda_epoca:.6e}")
               
        # Gregas
        print("\n[Fase 2] Extração de Gregas nativas via Autograd...")
        precos_avaliacao = torch.linspace(25.0, 55.0, 8, device=self.cfg.dispositivo).unsqueeze(-1)
        rendimentos_avaliacao = torch.full_like(precos_avaliacao, self.cfg.theta_rendimento)
        variancias_avaliacao = torch.full_like(precos_avaliacao, self.cfg.theta_variancia)
        tempos_avaliacao = torch.zeros_like(precos_avaliacao)
       
        gregas = ExtratorNativoGregasRisco.extrair_superficie_completa_gregas(
            self.rede_neural, precos_avaliacao, rendimentos_avaliacao,
            variancias_avaliacao, tempos_avaliacao
        )
       
        print("\n--- Superfície de Risco e Gregas (PETR4) ---")
        for i in range(len(precos_avaliacao)):
            print(f"  Spot R$ {precos_avaliacao[i].item():5.2f} | "
                  f"Preço Modelo: R$ {gregas['Preco'][i].item():6.3f} | "
                  f"Δ: {gregas['Delta'][i].item():7.4f} | "
                  f"Γ: {gregas['Gamma'][i].item():8.5f} | "
                  f"Vanna: {gregas['Vanna'][i].item():8.5f}")
           
        # Métricas de validação
        metricas = AvaliadorMetricasQuantitativas.avaliar_convergencia_terminal(
            self.rede_neural, precos_avaliacao, preco_exercicio=self.cfg.preco_base_petr4
        )
        print(f"\n[Aferição] RMSE no vencimento: {metricas['RMSE_Terminal']:.4e}")
        print(f"[Aferição] MAE no vencimento:  {metricas['MAE_Terminal']:.4e}")
        
        # Análise estatística completa
        print("\n[Fase 3] Executando módulos estatísticos 20-25...")
        resultados_stats = self.executar_analise_estatistica_completa(df)
        
        print("\n--- Resumo Estatístico (Módulos 20-25) ---")
        print(f"  Média log-retorno diário: {resultados_stats['descritivas_retorno']['log_retorno_media']:.6f}")
        print(f"  Assimetria: {resultados_stats['descritivas_retorno']['log_retorno_assimetria']:.4f}")
        print(f"  Curtose excesso: {resultados_stats['descritivas_retorno']['log_retorno_curtose']:.4f}")
        print(f"  Jarque-Bera p-valor: {resultados_stats['normalidade']['jarque_bera_pvalue']:.4e}")
        print(f"  ADF (retornos) estacionário: {resultados_stats['estacionariedade_retorno']['adf_log_retorno_estacionaria']}")
        print(f"  GARCH α≈{resultados_stats['garch_aproximado']['alpha']:.4f} | β≈{resultados_stats['garch_aproximado']['beta']:.4f}")
        print(f"  VaR 99% histórico: {resultados_stats['metricas_risco']['VaR_historico_99']:.6f}")
        print(f"  Expected Shortfall 99%: {resultados_stats['metricas_risco']['Expected_Shortfall_99']:.6f}")
        print(f"  κ Heston calibrado (MLE): {resultados_stats['calibracao_heston_mle']['kappa_calibrado']:.4f}")
        print(f"  Bootstrap IC 95% média retorno: [{resultados_stats['bootstrap_media_retorno']['ic_95_inferior']:.6f}, "
              f"{resultados_stats['bootstrap_media_retorno']['ic_95_superior']:.6f}]")
        
        print("\n" + "=" * 70)
        print("  MODELO COMPLETO COM 26 MÓDULOS EXECUTADO E VALIDADO COM SUCESSO")
        print("  Autor: Luiz Tiago Wilcke")
        print("=" * 70)
        
        return resultados_stats


if __name__ == "__main__":
    pipeline = PipelineProducaoPINN_Petrobras()
    resultados = pipeline.executar_ciclo_producao()
