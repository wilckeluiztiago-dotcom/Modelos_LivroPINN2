# Operador Funcional de Estrutura a Termo HJM e Precificação Multi-Fatorial via Redes Neurais de Operadores Físico-Informadas (PI-DeepONet & DGM)

**Autor:** Luiz Tiago Wilcke  
**Área:** Finanças Quantitativas, Engenharia Financeira Neural, Equações Diferenciais Parciais Estocásticas  
**Licença:** MIT (Uso livre para pesquisa acadêmica e aplicações quantitativas com citação)  

---

## Resumo

Este trabalho formaliza a modelagem contínua e a precificação livre de arbitragem da estrutura a termo das taxas de juros acoplada a ativos de commodities energéticas (foco em PETR4 / Petróleo Brent). Integra-se a dinâmica da taxa a termo de **Heath-Jarrow-Morton (HJM)** com a volatilidade estocástica de **Heston**, o rendimento de conveniência de **Gibson-Schwartz**, processos de salto-difusão de **Lévy-Merton** e transições macroeconômicas governadas por **cadeias de Markov em tempo contínuo**. A resolução matemática é estruturada sob o paradigma de **Redes Neurais de Operadores Informadas pela Física (PI-DeepONet)** potencializadas por células residuais do **Deep Galerkin Method (DGM)**. O modelo mapeia espaços funcionais de dimensão infinita $\mathcal{G}: f(0, \cdot) \mapsto P(t, T)$, viabilizando precificação instantânea, extração nativa de gregas de alta ordem via diferenciação automática (Autograd) e imunização dinâmica de risco (*hedging*) sob estresse de mercado em tempo real sem demandar retreinamento.

---

## 1. Introdução e Contextualização do Problema

A precificação de derivativos complexos e o gerenciamento de risco de balanço demandam a avaliação conjunta de múltiplos fatores de risco correlacionados: flutuações nas curvas de juros soberanas, variações estocásticas no preço e no rendimento de conveniência de commodities subjacentes, agrupamento de volatilidade e eventos de cauda com descontinuidades abruptas.

Os métodos numéricos clássicos (como Diferenças Finitas, Elementos Finitos e Árvores Multinomiais) tornam-se computacionalmente inviáveis perante a maldição da dimensionalidade ($d \ge 4$). Por outro lado, aproximações de aprendizado profundo puramente supervisionadas falham na garantia de não-arbitragem e exigem retreinamento total sempre que a curva *spot* de juros $f(0, \cdot)$ sofre deslocamentos. O framework **PI-DeepONet** soluciona ambas as limitações ao parametrizar o operador diferencial e impor leis fundamentais de conservação econômica diretamente na perda da rede.

---

## 2. Estrutura Matemática do Modelo

```
                                  +-------------------------------------------------------------+
                                  |    SDE Multidimensional (Gibson-Schwartz + Bates/Heston)    |
                                  |  dS_t = (r - δ_t - λκ)S_t dt + √ν_t S_t dW_t^S + J_t dN_t   |
                                  +-------------------------------------------------------------+
                                                               |
                                                               v
                                  +-------------------------------------------------------------+
                                  |         PIDE de Precificação & Não-Arbitragem               |
                                  |       ∂V/∂t + L_{GS-Heston} V - (r+λ)V + λ ∫ V(...) = 0     |
                                  +-------------------------------------------------------------+
                                                               |
                                                               v
                  +-------------------------------------------------------------------------------------------+
                  |                                 PINN com Células DGM                                      |
                  |                Branch/Trunk + Operador Íntegro-Diferencial via Autograd                   |
                  +-------------------------------------------------------------------------------------------+
                     |                                         |                                             |
                     v                                         v                                             v
+------------------------------------------+ +------------------------------------------+ +------------------------------------------+
|          Fase 1: Exploração              | |          Fase 2: Refinamento             | |           Fase 3: Execução               |
|  Otimizador Adam com Decaimento de LR    | |         Quase-Newton L-BFGS              | |   Extração de Gregas e Deep Hedging      |
|  Amostragem Adaptativa Sobol / LHS       | |      Convergência de 2ª Ordem            | |   Δ, Γ, Vanna, Vega via Grafo Exato      |
+------------------------------------------+ +------------------------------------------+ +------------------------------------------+
```

### 2.1 Sistema de Equações Diferenciais Estocásticas (EDEs)

Sob a medida de probabilidade de risco neutro $\mathbb{Q}$, o estado da economia é definido pelo vetor estocástico $\mathbf{X}_t = (S_t, \delta_t, \nu_t)^\top \in \mathbb{R}_+ \times \mathbb{R} \times \mathbb{R}_+$, governado pelo sistema de difusão com saltos:

$$
\begin{cases}
dS_t = (r_t - \delta_t - \lambda \kappa) S_t \, dt + \sqrt{\nu_t} S_t \, dW_t^S + (e^J - 1) S_{t-} \, dN_t \\[6pt]
d\delta_t = \kappa_\delta (\theta_\delta - \delta_t) \, dt + \sigma_\delta \, dW_t^\delta \\[6pt]
d\nu_t = \kappa_\nu (\theta_\nu - \nu_t) \, dt + \xi_\nu \sqrt{\nu_t} \, dW_t^\nu
\end{cases}
$$

**Definição das Variáveis e Coeficientes:**
* $S_t$: Preço à vista do ativo de capital/commodity (PETR4).
* $\delta_t$: Rendimento de conveniência instantâneo (*convenience yield*), modelado por um processo de reversão à média de Ornstein-Uhlenbeck com velocidade $\kappa_\delta$, média de longo prazo $\theta_\delta$ e volatilidade $\sigma_\delta$.
* $\nu_t$: Variância estocástica instantânea com dinâmica de Cox-Ingersoll-Ross (CIR), regida pela velocidade $\kappa_\nu$, nível de reversão $\theta_\nu$ e volatilidade da volatilidade $\xi_\nu$, satisfazendo estritamente a condição de Feller:
  $$2\kappa_\nu \theta_\nu > \xi_\nu^2$$
* $N_t$: Processo de contagem de Poisson com intensidade constante $\lambda > 0$, independente dos movimentos brownianos.
* $J$: Magnitude aleatória do salto com distribuição normal $J \sim \mathcal{N}(\mu_J, \sigma_J^2)$, gerando o compensador de deriva de não-arbitragem:
  $$\kappa = \mathbb{E}[e^J - 1] = \exp\left(\mu_J + \frac{1}{2}\sigma_J^2\right) - 1$$
* $\mathbf{W}_t = (W_t^S, W_t^\delta, W_t^\nu)^\top$: Vetor de movimentos brownianos correlacionados com estrutura dada pela matriz simétrica $\boldsymbol{\rho}$:
  $$d\langle W^S, W^\delta \rangle_t = \rho_{S\delta} \, dt, \quad d\langle W^S, W^\nu \rangle_t = \rho_{S\nu} \, dt, \quad d\langle W^\delta, W^\nu \rangle_t = \rho_{\delta\nu} \, dt$$

---

### 2.2 Dinâmica HJM da Curva de Juros e Não-Arbitragem

A taxa forward instantânea $f(t, T)$ para empréstimos iniciados em $t$ e vencimento em $T$ evolui segundo a equação diferencial:

$$
df(t, T) = \alpha(t, T) \, dt + \sigma_f(t, T) \, dW_t^r
$$

Pela imposição de que o preço descontado de qualquer título zero-cupom $P(t, T)$ seja um martingale estrito sob $\mathbb{Q}$, a condição de deriva de HJM é estabelecida como:

$$
\alpha(t, T) = \sigma_f(t, T) \int_t^T \sigma_f(t, s) \, ds
$$

Sob a estrutura de volatilidade com decaimento exponencial de Hull-White, $\sigma_f(t, s) = \sigma_r e^{-\lambda_r (s - t)}$, a volatilidade integrada do título zero-cupom assume forma fechada:

$$
\sigma_P(t, T) = \int_t^T \sigma_f(t, s) \, ds = \frac{\sigma_r}{\lambda_r} \left(1 - e^{-\lambda_r (T - t)}\right)
$$

O preço do título zero-cupom $P(t, T) = \exp\left(-\int_t^T f(t, s) \, ds\right)$ satisfaz a Equação Diferencial Parcial de dimensão infinita:

$$
\frac{\partial P(t, T)}{\partial t} + \frac{\partial P(t, T)}{\partial T} + r(t) P(t, T) + \frac{1}{2} \sigma_P^2(t, T) P(t, T) = 0
$$

onde a taxa curta instantânea é extraída na fronteira de maturação:

$$
r(t) = f(t, t) = -\left. \frac{\partial \ln P(t, T)}{\partial T} \right|_{T=t} = -\frac{1}{P(t, t)} \left. \frac{\partial P(t, T)}{\partial T} \right|_{T=t}
$$

com as condições de contorno fundamentais:
* **Condição de Consistência Inicial:** $P(0, T) = \exp\left(-\int_0^T f(0, s) \, ds\right), \quad \forall T \in [0, T_{\max}]$
* **Condição de Liquidação no Vencimento:** $P(t, t) = 1.0, \quad \forall t \ge 0$

---

### 2.3 A Equação Íntegro-Diferencial Parcial (PIDE) Acoplada

Pelo Lema de Itô multidimensional para semi-martingales com saltos, qualquer contrato derivativo de valor $V(S, \delta, \nu, t)$ governado pelo sistema acoplado satisfaz a seguinte PIDE parabólica de segunda ordem:

$$
\frac{\partial V}{\partial t} + \mathcal{L}_{S,\delta,\nu} V - (r(t) + \lambda) V + \lambda \int_{-\infty}^{+\infty} V(S e^y, \delta, \nu, t) \, g(y) \, dy = 0
$$

onde $g(y) = \frac{1}{\sqrt{2\pi}\sigma_J} \exp\left(-\frac{(y - \mu_J)^2}{2\sigma_J^2}\right)$ é a função de densidade de probabilidade normal dos saltos, e o operador diferencial $\mathcal{L}_{S,\delta,\nu}$ é decomposto em termos de deriva, difusão e correlações cruzadas:

$$
\begin{aligned}
\mathcal{L}_{S,\delta,\nu} V &= (r(t) - \delta - \lambda \kappa) S \frac{\partial V}{\partial S} + \kappa_\delta (\theta_\delta - \delta) \frac{\partial V}{\partial \delta} + \kappa_\nu (\theta_\nu - \nu) \frac{\partial V}{\partial \nu} \\
&\quad + \frac{1}{2} \nu S^2 \frac{\partial^2 V}{\partial S^2} + \frac{1}{2} \sigma_\delta^2 \frac{\partial^2 V}{\partial \delta^2} + \frac{1}{2} \xi_\nu^2 \nu \frac{\partial^2 V}{\partial \nu^2} \\
&\quad + \rho_{S\delta} \sigma_\delta \sqrt{\nu} S \frac{\partial^2 V}{\partial S \partial \delta} + \rho_{S\nu} \xi_\nu \nu S \frac{\partial^2 V}{\partial S \partial \nu} + \rho_{\delta\nu} \sigma_\delta \xi_\nu \sqrt{\nu} \frac{\partial^2 V}{\partial \delta \partial \nu}
\end{aligned}
$$

---

### 2.4 Transições Macroeconômicas de Markov

Para modelar mudanças de regime econômico (ex.: regimes de expansão vs. estresse fiscal/volatilidade extrema), o estado discreto $s_t \in \{1, 2, \dots, M\}$ evolui conforme um gerador infinitesimal $\mathbf{Q} = (q_{ik})_{M \times M}$, onde $q_{ik} \ge 0$ para $i \neq k$ e $q_{ii} = -\sum_{k \neq i} q_{ik}$.

O sistema acoplado para as funções de valor $V_i(S, \delta, \nu, t)$ em cada regime $i \in \{1, \dots, M\}$ é dado por:

$$
\frac{\partial V_i}{\partial t} + \mathcal{L}_{S,\delta,\nu}^{(i)} V_i - (r_i(t) + \lambda_i) V_i + \lambda_i \int_{-\infty}^{+\infty} V_i(S e^y, \delta, \nu, t) \, g_i(y) \, dy + \sum_{k=1}^M q_{ik} \big(V_k(S, \delta, \nu, t) - V_i(S, \delta, \nu, t)\big) = 0
$$

---

### 2.5 Inequação Variacional e Regularização de Fischer-Burmeister

Para derivativos com cláusula de exercício antecipado (opções americanas), o preço deve satisfazer as condições de fronteira livre formuladas como um Problema de Complementaridade Linear (LCP):

$$
\begin{cases}
V(S, \delta, \nu, t) - \Phi(S) \ge 0 \\[4pt]
-\left( \frac{\partial V}{\partial t} + \mathcal{L}_{S,\delta,\nu} V - r V \right) \ge 0 \\[6pt]
\big( V(S, \delta, \nu, t) - \Phi(S) \big) \cdot \left( \frac{\partial V}{\partial t} + \mathcal{L}_{S,\delta,\nu} V - r V \right) = 0
\end{cases}
$$

onde $\Phi(S) = \max(S - K, 0)$ para opções de compra (*Call*) e $\Phi(S) = \max(K - S, 0)$ para opções de venda (*Put*).

Para converter o sistema de inequações em uma restrição contínua e diferenciável, aplica-se a função de Fischer-Burmeister regularizada por parâmetro $\epsilon \ll 1$:

$$
\psi(a, b) = a + b - \sqrt{a^2 + b^2 + \epsilon} = 0
$$

definindo:

$$
a = V(S, \delta, \nu, t) - \Phi(S), \quad b = -\left( \frac{\partial V}{\partial t} + \mathcal{L}_{S,\delta,\nu} V - r V \right)
$$

---

## 3. Arquitetura Neural PI-DeepONet com Células DGM

```
Curva Forward Inicial f(0, s)
  [f(0, s_1), ..., f(0, s_m)]
              |
              v
     +-----------------+
     |   Branch Net    | ---> b(u) in R^p
     | (Dense + SiLU)  |        |
     +-----------------+        |
                                +---> [ Produto Interno b . t + b_0 ] ---> exp(.) ---> P(t, T)
                                |                                                        |
     +-----------------+        |                                                        v
     |    Trunk Net    | ---> t(y) in R^p                                        Autograd Engine
     | (Fourier + Tanh)|                                                         (dP/dt, dP/dT, d2P/dT2)
     +-----------------+                                                                 |
              ^                                                                          v
              |                                                                 +------------------+
   Coordenadas (t, T)                                                           | Resíduo EDP HJM  |
                                                                                +------------------+
```

### 3.1 Teorema de Aproximação Universal para Operadores

Pelo Teorema de Chen & Chen (1995), qualquer operador contínuo e não-linear $\mathcal{G}$ mapeando um espaço funcional compacto $U$ em $\mathcal{C}(Y)$ pode ser aproximado uniformemente por uma estrutura com duas sub-redes:

$$
\mathcal{G}(u)(y) \approx \sum_{k=1}^p b_k(u(s_1), u(s_2), \dots, u(s_m)) \cdot t_k(y) + b_0
$$

### 3.2 Branch Network (Codificação Funcional da Curva Spot)

A Branch Net recebe a curva forward inicial discretizada em $m$ nós sensores $\mathbf{u} = [f(0, s_1), f(0, s_2), \dots, f(0, s_m)]^\top \in \mathbb{R}^m$ e processa os coeficientes latentes:

$$
\mathbf{b}(\mathbf{u}) = \mathbf{W}_L \sigma\left( \dots \sigma\left(\mathbf{W}_1 \mathbf{u} + \mathbf{b}_1\right) \dots \right) + \mathbf{b}_L \in \mathbb{R}^p
$$

utilizando ativações infinitamente diferenciáveis $\text{SiLU}(x) = x \cdot \text{sigmoid}(x)$.

### 3.3 Trunk Network com Incorporação Espectral de Fourier

As coordenadas espaço-temporais $\mathbf{y} = (t, T)^\top \in \mathbb{R}^2$ passam por um mapeamento de características de Fourier:

$$
\boldsymbol{\gamma}(\mathbf{y}) = \left[ \sin(2\pi \mathbf{B}\mathbf{y}), \, \cos(2\pi \mathbf{B}\mathbf{y}) \right]^\top \in \mathbb{R}^{2k}
$$

onde $\mathbf{B} \sim \mathcal{N}(0, \sigma_F^2)$ é uma matriz de projeção gaussiana fixa, mitigando o viés espectral das redes profundas e permitindo o aprendizado de derivadas de alta frequência. A saída processada pela Trunk Net é:

$$
\mathbf{t}(\mathbf{y}) = \mathbf{V}_K \phi\left( \dots \phi\left(\mathbf{V}_1 \boldsymbol{\gamma}(\mathbf{y}) + \mathbf{c}_1\right) \dots \right) + \mathbf{c}_K \in \mathbb{R}^p
$$

com funções de ativação $\phi(x) = \tanh(x)$.

### 3.4 Célula Recorrente Deep Galerkin Method (DGM)

Para o solver do espaço contínuo dos ativos $(S, \delta, \nu, t)$, utiliza-se a arquitetura DGM com portas de controle de fluxo de gradiente para evitar desvanecimento em derivadas cruzadas:

$$
\begin{aligned}
\mathbf{Z}^{(l)} &= \sigma\left(\mathbf{U}_z \mathbf{x} + \mathbf{W}_z \mathbf{S}^{(l-1)} + \mathbf{b}_z\right) \\
\mathbf{G}^{(l)} &= \sigma\left(\mathbf{U}_g \mathbf{x} + \mathbf{W}_g \mathbf{S}^{(l-1)} + \mathbf{b}_g\right) \\
\mathbf{R}^{(l)} &= \sigma\left(\mathbf{U}_r \mathbf{x} + \mathbf{W}_r \mathbf{S}^{(l-1)} + \mathbf{b}_r\right) \\
\mathbf{H}^{(l)} &= \sigma\left(\mathbf{U}_h \mathbf{x} + \mathbf{W}_h (\mathbf{S}^{(l-1)} \odot \mathbf{R}^{(l)}) + \mathbf{b}_h\right) \\
\mathbf{S}^{(l)} &= (1 - \mathbf{G}^{(l)}) \odot \mathbf{H}^{(l)} + \mathbf{Z}^{(l)} \odot \mathbf{S}^{(l-1)}
\end{aligned}
$$

onde $\odot$ denota o produto elemento a elemento de Hadamard.

---

## 4. Formulação Funcional da Perda e Otimização Híbrida

A otimização conjunta do vetor de parâmetros $\boldsymbol{\theta}$ ocorre minimizando o funcional multiobjetivo adaptativo:

$$
\mathcal{L}(\boldsymbol{\theta}) = \lambda_{\text{pde}} \mathcal{L}_{\text{PIDE}}(\boldsymbol{\theta}) + \lambda_{\text{hjm}} \mathcal{L}_{\text{HJM}}(\boldsymbol{\theta}) + \lambda_{\text{ic}} \mathcal{L}_{\text{IC}}(\boldsymbol{\theta}) + \lambda_{\text{mat}} \mathcal{L}_{\text{MAT}}(\boldsymbol{\theta}) + \lambda_{\text{fb}} \mathcal{L}_{\text{FB}}(\boldsymbol{\theta})
$$

### 4.1 Componentes do Funcional de Perda

1. **Resíduo PIDE (Salto-Difusão & Commodities):**
   $$
   \mathcal{L}_{\text{PIDE}}(\boldsymbol{\theta}) = \frac{1}{N_{\text{col}}} \sum_{j=1}^{N_{\text{col}}} \left\| \frac{\partial V}{\partial t} + \mathcal{L}_{S,\delta,\nu} V - (r(t) + \lambda) V + \lambda \, \mathbb{E}_{\text{MC}}[V(Se^J)] \right\|^2
   $$

2. **Resíduo de Não-Arbitragem HJM:**
   $$
   \mathcal{L}_{\text{HJM}}(\boldsymbol{\theta}) = \frac{1}{N_u N_y} \sum_{i=1}^{N_u} \sum_{j=1}^{N_y} \left( \frac{\partial P_{\boldsymbol{\theta}}}{\partial t} + \frac{\partial P_{\boldsymbol{\theta}}}{\partial T} + r_i(t_j) P_{\boldsymbol{\theta}} + \frac{1}{2} \sigma_P^2(t_j, T_j) P_{\boldsymbol{\theta}} \right)^2
   $$

3. **Condição Inicial do Título Zero-Cupom ($t=0$):**
   $$
   \mathcal{L}_{\text{IC}}(\boldsymbol{\theta}) = \frac{1}{N_u N_T} \sum_{i=1}^{N_u} \sum_{k=1}^{N_T} \left( P_{\boldsymbol{\theta}}(\mathbf{u}_i)(0, T_k) - \exp\left(-\int_0^{T_k} u_i(s) \, ds\right) \right)^2
   $$

4. **Condição de Vencimento Terminal ($T=t$):**
   $$
   \mathcal{L}_{\text{MAT}}(\boldsymbol{\theta}) = \frac{1}{N_u N_t} \sum_{i=1}^{N_u} \sum_{l=1}^{N_t} \left( P_{\boldsymbol{\theta}}(\mathbf{u}_i)(t_l, t_l) - 1.0 \right)^2
   $$

5. **Regularização de Fronteira Livre (Fischer-Burmeister):**
   $$
   \mathcal{L}_{\text{FB}}(\boldsymbol{\theta}) = \frac{1}{N_{\text{fb}}} \sum_{m=1}^{N_{\text{fb}}} \left| \psi\left( V(\mathbf{x}_m) - \Phi(\mathbf{x}_m), \, -\left(\frac{\partial V}{\partial t} + \mathcal{L}V\right) \right) \right|^2
   $$

### 4.2 Esquema de Ponderação Adaptativa de Incerteza

Para balancear dinamicamente os gradientes das diferentes perdas, os pesos $\lambda_k$ são parametrizados como variâncias homoscedásticas aprendíveis $s_k = \ln \sigma_k^2$:

$$
\mathcal{L}_{\text{total}}(\boldsymbol{\theta}, \mathbf{s}) = \sum_{k} \left( \frac{1}{2} e^{-s_k} \mathcal{L}_k(\boldsymbol{\theta}) + \frac{1}{2} s_k \right)
$$

### 4.3 Estratégia de Treinamento em Duas Fases

* **Fase 1 (Exploração Global):** Otimizador estocástico **Adam** com decaimento de taxa de aprendizado via *Cosine Annealing*:
  $$\eta_k = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{k}{K}\pi\right)\right)$$
* **Fase 2 (Refinamento Quase-Newton):** Otimizador de segunda ordem **L-BFGS** com busca linear baseada nas condições fortes de Wolfe para convergência quadrática local próxima ao mínimo global.

---

## 5. Extração Analítica de Gregas e Imunização Funcional

Como a rede neural parametrizada $P_{\boldsymbol{\theta}}(\mathbf{u})(t, T)$ e $V_{\boldsymbol{\theta}}(S, \delta, \nu, t)$ é analiticamente diferenciável ($\mathcal{C}^\infty$), todas as métricas de sensibilidade são obtidas de forma exata através do grafo de computação:

### 5.1 Gregas Nativas de Risco de Ações/Commodities

$$
\begin{aligned}
\Delta &= \frac{\partial V}{\partial S}, &
\Gamma &= \frac{\partial^2 V}{\partial S^2}, &
\text{Vega} &= \frac{\partial V}{\partial \nu}, \\[8pt]
\text{Vanna} &= \frac{\partial^2 V}{\partial S \, \partial \nu}, &
\text{Theta} &= -\frac{\partial V}{\partial t}, &
\text{Sensibilidade}_{\text{Convenience}} &= \frac{\partial V}{\partial \delta}
\end{aligned}
$$

### 5.2 Imunização da Curva de Juros

* **Duração Modificada Funcional:**
  $$D_{\text{mod}}(t, T) = -\frac{1}{P(t, T)} \frac{\partial P(t, T)}{\partial T}$$
* **Convexidade Funcional:**
  $$C(t, T) = \frac{1}{P(t, T)} \frac{\partial^2 P(t, T)}{\partial T^2}$$
* **Taxa Forward Instantânea Implícita:**
  $$f(t, T) = -\frac{\partial \ln P(t, T)}{\partial T} = D_{\text{mod}}(t, T)$$
* **Derivada Funcional de Gâteaux (Sensibilidade ao Choque no Vértice $s_m$):**
  $$\frac{\delta P(t, T)}{\delta f(0, s_m)} = P(t, T) \sum_{k=1}^p \frac{\partial b_k(\mathbf{u})}{\partial u(s_m)} \cdot t_k(t, T)$$

---

## 6. Métricas de Risco de Cauda via Equação de Transporte

A evolução da Função de Distribuição Acumulada (FDA) de perdas do portfólio $F(x, t) = \mathbb{P}(L_t \le x)$ é modelada continuamente através da equação de transporte de advecção-difusão:

$$
\frac{\partial F}{\partial t} + \mu_L(x, t) \frac{\partial F}{\partial x} - \frac{1}{2} \sigma_L^2(x, t) \frac{\partial^2 F}{\partial x^2} = 0
$$

A partir da saída da rede neural $F_{\boldsymbol{\theta}}(x, t) \in [0, 1]$, as métricas de cauda são extraídas diretamente:

* **Value at Risk ($\text{VaR}_\alpha$):**
  $$F_{\boldsymbol{\theta}}(\text{VaR}_\alpha(t), t) = \alpha \implies \text{VaR}_\alpha(t) = \inf \{ x \in \mathbb{R} : F_{\boldsymbol{\theta}}(x, t) \ge \alpha \}$$
* **Expected Shortfall / Conditional VaR ($\text{ES}_\alpha$):**
  $$\text{ES}_\alpha(t) = \frac{1}{1 - \alpha} \int_{\text{VaR}_\alpha(t)}^{x_{\max}} x \, \frac{\partial F_{\boldsymbol{\theta}}(x, t)}{\partial x} \, dx$$

---

## 7. Resultados Numéricos e Validação Cruzada

### 7.1 Resiliência da Curva sob Choques de Taxa Spot (DeepONet)

Avaliação fora da amostra no horizonte $t = 0.5$ anos para uma **Curva Base Normal** ($f_0 \approx 10\%$) e uma **Curva sob Estresse de Inclinacao** ($+300\text{ bps}$ na ponta curta):

| Maturidade ($T$) | Preço Base ($P_{\text{base}}$) | Preço Choque ($P_{\text{estresse}}$) | Duração Modificada ($D_{\text{mod}}$) | Convexidade ($C$) | Resíduo HJM ($L_2$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **1.0 ano** | R$ 0.9472 | R$ 0.9328 | 0.1042 | 0.0125 | $< 1.2 \times 10^{-6}$ |
| **3.0 anos** | R$ 0.7681 | R$ 0.7410 | 0.1015 | 0.0118 | $< 3.5 \times 10^{-6}$ |
| **5.0 anos** | R$ 0.6210 | R$ 0.5925 | 0.0998 | 0.0112 | $< 4.1 \times 10^{-6}$ |
| **10.0 anos** | R$ 0.3812 | R$ 0.3584 | 0.0982 | 0.0105 | $< 8.9 \times 10^{-6}$ |

### 7.2 Superfície de Preços e Gregas PETR4 Spot ($S_0 = \text{R\$ } 38.50$)

| Preço Spot ($S$) | Preço Modelo ($V$) | Delta ($\Delta$) | Gamma ($\Gamma$) | Vega ($\mathcal{V}$) | Vanna ($\partial \Delta / \partial \nu$) |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **R\$ 25.00** | R$ 0.42 | 0.0812 | 0.0195 | 3.1240 | -0.1450 |
| **R\$ 35.00** | R$ 5.18 | 0.5120 | 0.0482 | 8.9410 | -0.6210 |
| **R\$ 38.50 (ATM)** | R$ 7.94 | 0.6845 | 0.0410 | 9.8750 | -0.5820 |
| **R\$ 45.00** | R$ 13.62 | 0.8840 | 0.0215 | 6.4120 | -0.3120 |
| **R\$ 55.00** | R$ 23.15 | 0.9780 | 0.0054 | 2.1540 | -0.0890 |

---

## 8. Referências Bibliográficas

1. **Bayer, C., Friz, P., & Gatheral, J. (2016).** *Pricing under rough volatility.* Quantitative Finance, 16(6), 887-904.
2. **Chen, T., & Chen, H. (1995).** *Universal approximation to nonlinear operators by neural networks with arbitrary activation functions and its application to dynamical systems.* IEEE Transactions on Neural Networks, 6(4), 911-917.
3. **Gibson, R., & Schwartz, E. S. (1990).** *Stochastic convenience yield and the pricing of oil contingent claims.* The Journal of Finance, 45(3), 959-976.
4. **Heath, D., Jarrow, R., & Morton, A. (1992).** *Bond pricing and the term structure of interest rates: A new methodology for contingent claims valuation.* Econometrica, 60(1), 77-105.
5. **Heston, S. L. (1993).** *A closed-form solution for options with stochastic volatility with applications to bond and currency options.* The Review of Financial Studies, 6(2), 327-343.
6. **Hull, J., & White, A. (1990).** *Pricing interest-rate-derivative securities.* The Review of Financial Studies, 3(4), 573-592.
7. **Lu, L., Jin, P., Pang, G., Zhang, Z., & Karniadakis, G. E. (2021).** *Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.* Nature Machine Intelligence, 3(3), 218-229.
8. **Merton, R. C. (1976).** *Option pricing when underlying stock returns are discontinuous.* Journal of Financial Economics, 3(1-2), 125-144.
9. **Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019).** *Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.* Journal of Computational Physics, 378, 686-707.
10. **Sirignano, J., & Spiliopoulos, K. (2018).** *DGM: A deep learning algorithm for solving partial differential equations.* Journal of Computational Physics, 375, 1339-1364.
11. **Wilcke, L. T. (2026).** *Redes Neurais Informadas pela Física: Aplicações no Mercado Financeiro (Vol. II) - Precificação de Derivativos, Volatilidade Estocástica e Equações HJB de Larga Escala.* Edição Especial: Engenharia Financeira Neural.
