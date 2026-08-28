# PI-DeepONet-HJM  
**Operador Neural Física-Informado para Dinâmica da Curva de Juros (Heath-Jarrow-Morton)**

**Autor:** Luiz Tiago Wilcke  
**Baseado em:** *Redes Neurais Informadas pela Física – Volume II* (Edição Especial: Precificação de Derivativos, Volatilidade Estocástica e Equações HJB de Larga Escala)

---

## 1. Visão Geral

Este repositório implementa um **Physics-Informed Deep Operator Network (PI-DeepONet)** capaz de realizar o mapeamento funcional instantâneo:

$$
\mathcal{G}: f(0,\cdot) \;\longmapsto\; P(t,T)
$$

isto é, recebe a **curva forward inteira inicial** \(f(0,\cdot)\) (observada em \(m\) sensores) e gera a **superfície completa de preços de títulos zero-cupom** \(P(t,T)\).

A rede é treinada de forma **não-supervisionada / semi-supervisionada** impondo:

1. A **condição de livre-arbitragem de Heath-Jarrow-Morton**;
2. A **EDP de precificação do título zero-cupom**;
3. A **condição inicial** \(P(0,T)=\exp\bigl(-\int_0^T f(0,s)\,ds\bigr)\).

---

## 2. Fundamentos Matemáticos

### 2.1 Dinâmica HJM da Curva Forward

Sob a medida neutra ao risco a curva forward instantânea satisfaz a SDE:

$$
\mathrm{d}f(t,T)=\alpha(t,T)\,\mathrm{d}t+\sigma(t,T)\,\mathrm{d}W_t
$$

### 2.2 Condição de Livre-Arbitragem de Heath-Jarrow-Morton

A ausência de arbitragem força o drift a ser completamente determinado pela volatilidade:

$$
\alpha(t,T)=\sigma(t,T)\int_t^T\sigma(t,s)\,\mathrm{d}s
$$

### 2.3 Preço do Título Zero-Cupom

$$
P(t,T)=\exp\left(-\int_t^T f(t,s)\,\mathrm{d}s\right)
$$

e a taxa curta é o limite

$$
r(t)=f(t,t).
$$

### 2.4 Equação Diferencial Parcial de Precificação

Aplicando o Lema de Itô ao preço do título obtém-se a EDP linear:

$$
\frac{\partial P}{\partial t}+r(t)P-f(t,T)P+\frac12\sigma_P^2(t,T)P=0,
$$

onde a volatilidade do título é

$$
\sigma_P(t,T)=-\int_t^T\sigma(t,s)\,\mathrm{d}s.
$$

### 2.5 Arquitetura PI-DeepONet

- **Branch Net** \(b(\cdot)\): processa o vetor de sensores  
  $$
  u=\bigl[f(0,T_1),\dots,f(0,T_m)\bigr]\in\mathbb{R}^m
  $$
  e produz o vetor de coeficientes \(b(u)\in\mathbb{R}^p\).

- **Trunk Net** \(t(\cdot)\): processa as coordenadas de consulta  
  $$
  y=(t,T)\in\mathbb{R}^2
  $$
  e produz as funções de base \(t(y)\in\mathbb{R}^p\).

- **Saída do operador** (garantindo positividade):

$$
P(t,T)=\exp\left(\sum_{k=1}^p b_k(u)\,t_k(y)+b_0\right).
$$

A perda física impõe o residual da dinâmica HJM e da EDP **diretamente sobre o produto interno** \(\langle b(u),t(y)\rangle\).

---

## 3. Estrutura do Repositório (21 Módulos)

```
PI-DeepONet-HJM/
├── src/
│   ├── config.py (Módulo 01)              # Hiperparâmetros e configuração global
│   ├── utils.py (Módulo 02)               # Utilitários de tensor e normalização
│   ├── matematica_hjm.py (Módulo 03)      # Drift de livre-arbitragem, σ_P, etc.
│   ├── geracao_curvas.py (Módulo 04)      # Ensemble de curvas Nelson-Siegel
│   ├── rede_branch.py (Módulo 05)         # Branch Net
│   ├── rede_trunk.py (Módulo 06)          # Trunk Net
│   ├── arquitetura_deeponet.py (Módulo 07)# PI-DeepONet completa
│   ├── residual_hjm.py (Módulo 08)        # Residual da condição HJM
│   ├── residual_pde.py (Módulo 09)        # Residual da EDP de precificação
│   ├── perda_composta.py (Módulo 10)      # Loss física + dados + livre-arbitragem
│   ├── amostragem.py (Módulo 11)          # Latin Hypercube / amostragem de domínio
│   ├── otimizacao.py (Módulo 12)          # Adam + L-BFGS
│   ├── treinamento.py (Módulo 13)         # Loop híbrido de treinamento
│   ├── avaliacao.py (Módulo 14)           # Métricas e superfície
│   ├── visualizacao.py (Módulo 15)        # Plots 3-D e curvas
│   ├── extracao_gregas.py (Módulo 16)     # Duration, convexidade via autograd
│   ├── calibracao.py (Módulo 17)          # Calibração de volatilidade
│   ├── logger.py (Módulo 18)              # Logging JSON
│   ├── main.py (Módulo 19)                # Script principal
│   ├── exportacao.py (Módulo 20)          # Salvar / carregar checkpoint
│   └── pipeline.py (Módulo 21)            # Pipeline de inferência em produção
├── results/                      # Figuras e checkpoints
├── requirements.txt
└── README.md
```

---

## 4. Instalação

```bash
git clone https://github.com/seu-usuario/PI-DeepONet-HJM.git
cd PI-DeepONet-HJM
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
pip install -r requirements.txt
```

---

## 5. Uso Rápido

```python
from src import PipelineHJM, CONFIG
from src.04_geracao_curvas import curva_forward_nelson_siegel
import torch

# Carrega modelo treinado (ou instancia um novo)
pipe = PipelineHJM("results/modelo_pi_deeponet_hjm.pt")

# Curva forward de exemplo (Nelson-Siegel)
T_sens = torch.tensor(CONFIG.maturidades_sensores)
f0 = curva_forward_nelson_siegel(T_sens)

# Superfície completa P(t,T)
superficie = pipe.predizer_superficie(f0)
print(superficie["P"].shape)   # (resolucao, resolucao)

# Preço pontual
preco = pipe.predizer_preco(f0.numpy(), t=1.0, T=5.0)
print(f"P(1,5) = {preco:.6f}")
```

### Treinar do zero

```bash
python -m src.19_main
```

---

## 6. Função de Perda Composta

$$
\mathcal{L}=\lambda_{\text{fis}}\underbrace{\bigl\|R_{\text{EDP}}\bigr\|_{L^2}^2}_{\text{residual da EDP}}
+\lambda_{\text{dados}}\underbrace{\bigl\|P(0,T)-P_0^{\text{exato}}\bigr\|_{L^2}^2}_{\text{condição inicial}}
+\lambda_{\text{LA}}\underbrace{\bigl\|R_{\text{HJM}}\bigr\|_{L^2}^2}_{\text{livre-arbitragem}}
$$

com

$$
R_{\text{EDP}}=\frac{\partial P}{\partial t}+rP-fP+\frac12\sigma_P^2P.
$$

Todos os gradientes são obtidos por **diferenciação automática** (PyTorch Autograd).

---

## 7. Variáveis em Português (Glossário)

| Símbolo matemático | Variável no código          | Significado                              |
|--------------------|-----------------------------|------------------------------------------|
| \(f(t,T)\)         | `taxa_forward` / `f_tT`     | Taxa forward instantânea                 |
| \(P(t,T)\)         | `preco_titulo` / `P`        | Preço do título zero-cupom               |
| \(r(t)\)           | `taxa_curta` / `r`          | Taxa de juros instantânea                |
| \(\sigma(t,T)\)    | `volatilidade_hjm`          | Volatilidade da curva forward            |
| \(\sigma_P(t,T)\)  | `volatilidade_preco_titulo` | Volatilidade do preço do título          |
| \(\alpha(t,T)\)    | `drift_livre_arbitragem`    | Drift de livre-arbitragem                |
| \(u\)              | `u` / `curva_forward`       | Vetor de sensores da curva inicial       |
| \(b(u)\)           | saída da `RedeBranch`       | Coeficientes da Branch Net               |
| \(t(y)\)           | saída da `RedeTrunk`        | Funções de base da Trunk Net             |

---

## 8. Referências

- Heath, Jarrow & Morton (1992) – Bond Pricing and the Term Structure of Interest Rates  
- Lu et al. (2021) – Learning nonlinear operators via DeepONet  
- Raissi, Perdikaris & Karniadakis (2019) – Physics-informed neural networks  
- Wilcke, L. T. – *Redes Neurais Informadas pela Física*, Volume II (2024/2025)

---

## 9. Licença e Citação

```bibtex
@software{wilcke2025pideeponethjm,
  author = {Wilcke, Luiz Tiago},
  title  = {PI-DeepONet-HJM: Operador Neural Física-Informado para a Dinâmica da Curva de Juros},
  year   = {2025},
  url    = {https://github.com/seu-usuario/PI-DeepONet-HJM}
}
```

---

**Luiz Tiago Wilcke** – Bacharel em Estatística  
*VOLUME II – EDIÇÃO ESPECIAL: ENGENHARIA FINANCEIRA NEURAL*
