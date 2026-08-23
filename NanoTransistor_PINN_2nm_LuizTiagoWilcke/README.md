# Modelagem Completa de Nanotransistor de Silício de 2 nm com Dopagem de Fósforo via Redes Neurais Informadas pela Física (PINNs)

**Autor:** Luiz Tiago Wilcke  
**Afiliação:** Bacharel em Estatística | Engenharia Financeira Neural (adaptação metodológica)  
**Data:** Agosto 2026  
**Licença:** MIT  

## Visão Geral

Este repositório apresenta a **modelagem completa e modular** de um nanotransistor de silício (Si) com canal de ~2 nm (tecnologia N2 / Gate-All-Around Nanosheet ou Nanowire FET) dopado com fósforo (P), utilizando o paradigma de **Physics-Informed Neural Networks (PINNs)** inspirado no livro *Redes Neurais Informadas pela Física – Aplicações no Mercado Financeiro* (Volume II, Edição Especial) de Luiz Tiago Wilcke.

A metodologia de PINNs, originalmente desenvolvida para EDPs financeiras (Black-Scholes, Heston, HJB, Fokker-Planck, etc.), é transferida para a física de semicondutores nanoescala, resolvendo de forma mesh-free e diferenciável as equações de:

- Poisson (eletrostática)
- Schrödinger (confinamento quântico)
- Continuidade / Drift-Diffusion com correções quânticas
- Transporte balístico / NEGF simplificado via densidades de estados

O projeto contém **mais de 30 módulos complexos** organizados em classes Python, com variáveis em português, amostragem Latin Hypercube, otimização híbrida Adam + L-BFGS, extração automática de “Gregas” (sensibilidades) via Autograd, e validação contra parâmetros reais da literatura de 2 nm node.

## Equações Governantes (Renderizadas para GitHub)

### 1. Equação de Poisson (Eletrostática)

$$
\nabla \cdot \big( \varepsilon(\mathbf{r}) \nabla \phi(\mathbf{r}) \big) = -\rho(\mathbf{r}) = -q \big[ p(\mathbf{r}) - n(\mathbf{r}) + N_D^+(\mathbf{r}) - N_A^-(\mathbf{r}) \big]
$$

onde $\phi$ é o potencial eletrostático, $\varepsilon$ a permissividade, $n,p$ densidades de elétrons/buracos, $N_D^+$ concentração de doadores ionizados (fósforo), $N_A^-$ aceitadores.

### 2. Equação de Schrödinger (Efetiva-Massa, 1D/2D)

$$
\left[ -\frac{\hbar^{2}}{2m^{*}} \nabla^{2} + V(\mathbf{r}) \right] \psi_{i}(\mathbf{r}) = E_{i}\, \psi_{i}(\mathbf{r})
$$

com potencial efetivo $V(\mathbf{r}) = -q\phi(\mathbf{r}) + \Delta E_{c}(\mathbf{r})$ e densidade eletrônica:

$$
n(\mathbf{r}) = \sum_{i} |\psi_{i}(\mathbf{r})|^{2}\, f(E_{i}; E_{F})
$$

### 3. Equações de Continuidade (Drift-Diffusion + Quantum)

$$
\frac{\partial n}{\partial t} = \frac{1}{q} \nabla \cdot \mathbf{J}_{n} + G - R
$$

$$
\mathbf{J}_{n} = q\mu_{n} n \mathbf{E} + q D_{n} \nabla n + \mathbf{J}_{\text{quantum}}
$$

### 4. Perda Composta da PINN (Analogia Financeira → Física de Dispositivos)

$$
\mathcal{L} = \lambda_{\text{Poisson}}\,\mathcal{L}_{\text{Poisson}} + \lambda_{\text{Schr}}\,\mathcal{L}_{\text{Schrödinger}} + \lambda_{\text{BC}}\,\mathcal{L}_{\text{contorno}} + \lambda_{\text{dados}}\,\mathcal{L}_{\text{dados}}
$$

onde cada termo residual é calculado via diferenciação automática (Autograd) exatamente como nas PINNs de Black-Scholes e Heston do livro-base.

### 5. Perfil de Dopagem de Fósforo (Realista)

$$
N_{D}(x) = N_{\text{S/D}} \exp\left( -\frac{(x - x_{\text{S/D}})^{2}}{2\sigma^{2}} \right) + N_{\text{canal}}
$$

com $N_{\text{S/D}} \approx 2 \times 10^{20}\,\text{cm}^{-3}$ e $N_{\text{canal}} \approx 1 \times 10^{15} - 2 \times 10^{17}\,\text{cm}^{-3}$ (valores típicos do nó N2).

## Estrutura do Projeto (30+ Módulos)

```text
NanoTransistor_PINN_2nm_LuizTiagoWilcke/
├── README.md
├── requirements.txt
├── src/
│   ├── geometria_dispositivo.py              # Definição GAA / Double-Gate 2 nm
│   ├── parametros_materiais_si.py             # Si, ε, m*, Eg, μ
│   ├── perfil_dopagem_fosforo.py              # N_D(x) gaussiano / erfc
│   ├── equacao_poisson.py                     # Residual Poisson
│   ├── equacao_schrodinger.py                 # Residual Schrödinger + autovalores
│   ├── arquitetura_pinn_poderosa.py           # MLP residual + Fourier features
│   ├── main_treinamento.py                    # Script principal de treinamento
│   ├── modulo_06.py ... modulo_30.py          # 25 módulos avançados de extensão
│   │     (continuidade, BC, LHS, otimização híbrida, Gregas, transporte
│   │      balístico, NEGF, self-consistent, calibração inversa, RDF,
│   │      temperatura, mobilidade, SRH/Auger, tunneling, GAA 3D, FNO,
│   │      DeepONet, MFG, risco sistêmico, rugosidade, HJB layout,
│   │      Bayesian PINN, solver de produção)
├── data/                                      # Parâmetros reais N2, doping P
├── results/                                   # Id-Vg, potencial, densidade, loss
├── figures/                                   # Gráficos gerados
├── docs/
│   └── artigo_cientifico.md
└── notebooks/
```

## Parâmetros Físicos Reais Utilizados (N2 Node)

| Parâmetro                    | Valor                                      | Fonte / Observação                  |
|-----------------------------|--------------------------------------------|-------------------------------------|
| Espessura de canal ($t_{\mathrm{Si}}$) | 2.0 – 2.8 nm                       | Nanosheet thickness (lit. 2024)    |
| Comprimento de porta ($L_g$) | 14 nm                                     | Contacted Gate Pitch 45 nm         |
| Dopagem S/D (P)             | $2\times10^{20}$ – $4\times10^{20}$ cm$^{-3}$ | In-situ epitaxial P            |
| Dopagem canal               | $1\times10^{15}$ – $2\times10^{17}$ cm$^{-3}$ | Baixa para evitar RDF          |
| EOT                         | 1.32 nm                                    | High-k stack                       |
| Massa efetiva elétron       | $0.26\,m_0$ (transversal)                  | Si $\langle 100\rangle$            |
| Permissividade Si           | $11.7\,\varepsilon_0$                      | —                                  |
| Temperatura                 | 300 K                                      | Ambiente                           |

## Como Executar

```bash
pip install -r requirements.txt
cd src
python main_treinamento.py --modo self_consistent --epochs_adam 5000 --epochs_lbfgs 500
```

## Resultados Numéricos Principais

- Convergência da perda física < 10⁻⁵ em ~ few thousand epochs  
- Potencial eletrostático e densidade de elétrons auto-consistentes  
- Curva I_D–V_G com SS ≈ 65–70 mV/dec e I_ON/I_OFF > 10⁵  
- Extração de capacitâncias e “Gregas” via Autograd em tempo real  

## Artigo Científico

Veja `docs/artigo_cientifico.md` para a descrição completa da metodologia, validação e discussões.

## Citação

```bibtex
@software{wilcke2026nanotransistor,
  author = {Wilcke, Luiz Tiago},
  title  = {Modelagem Completa de Nanotransistor de 2 nm via PINNs},
  year   = {2026},
  url    = {https://github.com/...}
}
```

---

**Inspirado no framework de PINNs financeiras do autor e adaptado para física de dispositivos nanoeletrônicos.**  
*“Entender o Universo — da Black-Scholes ao Schrödinger-Poisson.”*
