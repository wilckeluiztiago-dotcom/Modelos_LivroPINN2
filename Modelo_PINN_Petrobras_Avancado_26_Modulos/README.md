# Modelo PINN Deep Galerkin Multi-Fatorial Avançado — Petrobras (PETR4)

**Autor:** Luiz Tiago Wilcke  
**Versão:** 2.0 (26 Módulos)  
**Linguagem:** Python 3.10+  
**Frameworks principais:** PyTorch, SciPy, Statsmodels, Pandas, NumPy  

---

## 1. Visão Geral

Este repositório contém um **modelo de precificação de opções americanas/europeias sobre PETR4** (Petrobras) baseado em **Physics-Informed Neural Networks (PINN)** com arquitetura **Deep Galerkin Method (DGM)**.

O modelo resolve numericamente a **Equação Diferencial Parcial Integro-Diferencial (PIDE)** multi-fatorial que governa o preço de um derivativo sob:

- Processo de **Gibson-Schwartz** (rendimento de conveniência do petróleo);
- Processo de **Heston** (volatilidade estocástica);
- **Saltos de Merton** (processo de Lévy composto);
- **Mudança de regime de Markov** (dois estados: normal e estresse).

Além da resolução da PIDE via PINN, o modelo incorpora **26 módulos**, dos quais os módulos 20–25 utilizam bibliotecas estatísticas sofisticadas (`scipy.stats`, `statsmodels`, `pandas`) para calibração, testes de hipótese, métricas de risco e análise de sensibilidade.

---

## 2. Estrutura dos 26 Módulos

| Módulo | Nome | Função Principal |
|--------|------|------------------|
| 01 | Configurações e Hiperparâmetros Globais | Dataclass central de todos os parâmetros de mercado e de otimização |
| 02 | Motor de Dados e Tensores Financeiros | Geração de séries sintéticas e conversão para DataFrame |
| 03 | Simulador Monte Carlo Multi-Fatorial | Simulação de trajetórias correlacionadas (Cholesky) |
| 04 | Amostrador de Colocalização (Sobol) | Quasi-Monte Carlo no hipercubo unitário |
| 05 | Incorporação Espectral de Fourier | Feature embedding senoidal/cossenoidal |
| 06 | Célula Recorrente DGM | Unidade básica do Deep Galerkin Method |
| 07 | Arquitetura Principal PINN-DGM | Rede neural com duas cabeças (regimes de Markov) |
| 08 | Motor Autograd de Alta Ordem | Derivadas de 1ª e 2ª ordem + cruzadas via `torch.autograd` |
| 09 | Resíduo EDP Gibson-Schwartz | Operador diferencial do rendimento de conveniência |
| 10 | Operador Íntegro-Diferencial de Saltos Merton | Integração de Monte Carlo do termo de salto |
| 11 | Resíduo EDP Heston | Operador de volatilidade estocástica |
| 12 | Acoplador de Transição de Regime Markov | Termos de acoplamento entre regimes |
| 13 | Regularizador Fischer-Burmeister | Condição de complementaridade da fronteira livre |
| 14 | Calibrador Inverso de Parâmetros Latentes | Estimação de $\kappa$ via gradiente |
| 15 | Agregador de Perdas Multi-Objetivo Adaptativo | Pesos aprendíveis (uncertainty weighting) |
| 16 | Motor de Otimização Híbrida (Adam + L-BFGS) | Treinamento em duas fases |
| 17 | Extrator Nativo de Gregas | Delta, Gamma, Vega, Vanna, Theta via Autograd |
| 18 | Motor de Cobertura Dinâmica (Deep Hedging) | Rede de decisão de hedge com custos de transação |
| 19 | Avaliador de Métricas Quantitativas | RMSE/MAE no vencimento |
| 20 | Análise Estatística Descritiva e Inferencial | `scipy.stats` + pandas (assimetria, curtose, JB, Shapiro, correlações) |
| 21 | Diagnóstico de Estacionariedade | ADF, KPSS, Ljung-Box, Breusch-Pagan (`statsmodels`) |
| 22 | Estimador de Volatilidade Realizada e GARCH | Vol realizada rolling + aproximação GARCH(1,1) via OLS |
| 23 | Motor de Métricas de Risco Avançadas | VaR histórico/paramétrico, Expected Shortfall, t-Student |
| 24 | Calibrador Estatístico Avançado (MLE) | Maximum Likelihood + método dos momentos (`scipy.optimize`) |
| 25 | Análise de Sensibilidade e Bootstrap | Bootstrap não-paramétrico + elasticidades locais |
| 26 | Pipeline Integrado de Produção | Orquestrador master que executa todos os módulos |

---

## 3. Equações Fundamentais do Modelo

### 3.1 Processo Estocástico Multi-Fatorial

O preço à vista $S_t$, o rendimento de conveniência $\delta_t$ e a variância $v_t$ seguem o sistema:

$$\begin{aligned} \frac{dS_t}{S_t} &= (r - \delta_t - \lambda \kappa)\, dt + \sqrt{v_t}\, dW_t^{S} + (e^{J}-1)\, dN_t, \\ d\delta_t &= \kappa_{\delta}(\theta_{\delta} - \delta_t)\, dt + \sigma_{\delta}\, dW_t^{\delta}, \\ dv_t &= \kappa_{v}(\theta_{v} - v_t)\, dt + \xi\sqrt{v_t}\, dW_t^{v}. \end{aligned}$$

**Onde:**

- $r$ = taxa livre de risco (Selic);
- $\lambda$ = intensidade do processo de Poisson $N_t$;
- $J \sim \mathcal{N}(\mu_J, \sigma_J^{2})$ = tamanho do salto log-normal;
- $\kappa = \mathbb{E}[e^{J}-1]$ = compensador de salto;
- $W^{S},\; W^{\delta},\; W^{v}$ = movimentos brownianos correlacionados com matriz de correlação $\boldsymbol{\rho}$.

---

### 3.2 Equação PIDE com Mudança de Regime

Seja $V_{i}(S,\delta,v,t)$ o preço do derivativo no regime $i \in \{1,2\}$.  
A PIDE acoplada é formulada como:

$$\begin{aligned} \frac{\partial V_{i}}{\partial t} &+ (r - \delta - \lambda\kappa) S \frac{\partial V_{i}}{\partial S} + \kappa_{\delta}(\theta_{\delta}-\delta) \frac{\partial V_{i}}{\partial \delta} + \kappa_{v}(\theta_{v}-v) \frac{\partial V_{i}}{\partial v} \\ &+ \frac{1}{2} v S^{2} \frac{\partial^{2} V_{i}}{\partial S^{2}} + \frac{1}{2} \sigma_{\delta}^{2} \frac{\partial^{2} V_{i}}{\partial \delta^{2}} + \frac{1}{2} \xi^{2} v \frac{\partial^{2} V_{i}}{\partial v^{2}} \\ &+ \rho_{S\delta} \sigma_{\delta} S\sqrt{v} \frac{\partial^{2} V_{i}}{\partial S \partial\delta} + \rho_{Sv} \xi S v \frac{\partial^{2} V_{i}}{\partial S \partial v} + \rho_{\delta v} \sigma_{\delta} \xi\sqrt{v} \frac{\partial^{2} V_{i}}{\partial \delta \partial v} \\ &+ \lambda \int_{-\infty}^{\infty} \Bigl( V_{i}(S e^{z},\delta,v,t) - V_{i}(S,\delta,v,t) \Bigr) \nu(dz) + q_{ij}(V_{j} - V_{i}) - r V_{i} = 0 \end{aligned}$$

**Taxas de transição de Markov:**

$$q_{12} = \lambda_{1\to 2}, \qquad q_{21} = \lambda_{2\to 1}$$

---

### 3.3 Condição Terminal e Fronteira Livre (Opção Americana)

No vencimento $t = T$:

$$V_{i}(S,\delta,v,T) = \max(S - K,\, 0)$$

Para opções americanas, impõe-se a condição de complementaridade de **Fischer-Burmeister**:

$$\Phi_{\text{FB}}(a,b) = a + b - \sqrt{a^{2} + b^{2} + \varepsilon} = 0$$

onde:

$$a = V - (S-K)^{+}, \qquad b = -\mathcal{L}V$$

e $\mathcal{L}$ denota o operador infinitesimal da PIDE.

---

### 3.4 Função de Perda da PINN

A perda total é a combinação ponderada adaptativa (*uncertainty weighting*):

$$\mathcal{L} = e^{-s_{\text{PDE}}}\mathcal{L}_{\text{PDE}} + s_{\text{PDE}} + e^{-s_{\text{BC}}}\mathcal{L}_{\text{BC}} + s_{\text{BC}} + e^{-s_{\text{FB}}}\mathcal{L}_{\text{FB}} + s_{\text{FB}}$$

Os pesos $s_{\bullet}$ são parâmetros aprendíveis durante a retropropagação.

---

### 3.5 Gregas via Diferenciação Automática

$$\begin{aligned} \Delta &= \frac{\partial V}{\partial S}, & \Gamma &= \frac{\partial^{2} V}{\partial S^{2}}, & \mathcal{V} &= \frac{\partial V}{\partial v}, \\ \text{Vanna} &= \frac{\partial^{2} V}{\partial S\,\partial v}, & \Theta &= -\frac{\partial V}{\partial t} \end{aligned}$$

---

### 3.6 Métricas de Risco (Módulo 23)

**Value-at-Risk** (nível de confiança $\alpha$):

$$\text{VaR}_{\alpha} = \inf\bigl\{ x \in \mathbb{R} : P(L > x) \le 1-\alpha \bigr\}$$

**Expected Shortfall** (Conditional VaR):

$$\text{ES}_{\alpha} = \mathbb{E}\bigl[ L \mid L \ge \text{VaR}_{\alpha} \bigr]$$

---

## 4. Bibliotecas Estatísticas Utilizadas

| Biblioteca | Uso principal nos módulos |
|------------|---------------------------|
| `scipy.stats` | Jarque-Bera, Shapiro-Wilk, Anderson-Darling, KDE, t-Student, momentos, correlações |
| `scipy.optimize` | Maximização da verossimilhança (L-BFGS-B) |
| `statsmodels` | ADF, KPSS, ACF/PACF, Ljung-Box, Breusch-Pagan, OLS |
| `pandas` | Manipulação de séries temporais e DataFrames |
| `numpy` | Álgebra linear, bootstrap, percentis |

---

## 5. Como Executar

```bash
# Requisitos
pip install torch numpy pandas scipy statsmodels matplotlib

# Execução
python modelo_pinn_petrobras_26_modulos.py
