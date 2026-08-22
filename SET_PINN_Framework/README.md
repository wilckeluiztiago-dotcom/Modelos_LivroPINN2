# Framework PINN para Transporte Quântico em Transistor de Elétron Único (SET)

**Autor:** Luiz Tiago Wilcke  
**Base teórica:** Volume II – *Redes Neurais Informadas pela Física: Aplicações no Mercado Financeiro* (transposto para Física da Matéria Condensada e Nanoeletrônica)

---

## Visão Geral

Este framework implementa **32 módulos** de Redes Neurais Informadas pela Física (PINNs) para a modelagem rigorosa do transporte quântico em um **Single-Electron Transistor (SET)** sob a Teoria Ortodoxa de Tunelamento de Elétrons Únicos.

O código cobre:

- Energia livre de Gibbs e elipses de estabilidade de Coulomb  
- Taxas de tunelamento Fermi-Dirac  
- Equação mestra de estados de carga  
- Equação de Fokker-Planck contínua  
- Acoplamento Schrödinger-Poisson  
- Operadores fracionários (Caputo / rough noise)  
- DeepONet e Fourier Neural Operators  
- Deep Galerkin Method (DGM)  
- Calibração inversa de parâmetros do dispositivo  
- Quantificação de incerteza bayesiana  
- Extração de Gregas quânticas (transcondutância, etc.)  
- Simulação Monte Carlo (Gillespie) de referência  
- Geração de mapas de Diamantes de Coulomb  

Todo o código utiliza **PyTorch** com precisão `float64`, tipagem estrita, docstrings matemáticas e identificadores em **português**.

---

## Estrutura dos 32 Módulos

| #  | Arquivo                              | Descrição |
|----|--------------------------------------|-----------|
| 01 | `constantes_fisicas.py`              | Constantes SI (e, ħ, k_B, m₀, ε₀) |
| 02 | `configuracao_dispositivo.py`        | Capacitâncias, resistências e temperatura do SET |
| 03 | `energia_livre_eletrostatica.py`     | ΔF± e superfície de estabilidade |
| 04 | `estatistica_fermi_dirac.py`         | Distribuição de Fermi-Dirac regularizada |
| 05 | `taxas_tunelamento.py`               | Taxas Γ_S±, Γ_D± |
| 06 | `amostragem_hipercubo_latino.py`     | Latin Hypercube Sampling (LHS) |
| 07 | `redes_base_ativacoes.py`            | MLP + Tanh / Swish / SIREN |
| 08 | `celula_dgm_transporte.py`           | Célula Deep Galerkin Method |
| 09 | `operador_autograd_quantico.py`      | Derivadas de 1ª e 2ª ordem via autograd |
| 10 | `pinn_equacao_mestre.py`             | PINN da equação mestra discreta |
| 11 | `pinn_fokker_planck_continua.py`     | PINN da Fokker-Planck de carga |
| 12 | `regularizacao_fischer_burmeister.py`| Complementaridade de Coulomb |
| 13 | `solver_schrodinger_poisson.py`      | Acoplamento Schrödinger-Poisson |
| 14 | `camada_convolucional_fourier.py`    | SpectralConv (FNO) |
| 15 | `deeponet_operador_transporte.py`    | DeepONet de condutância |
| 16 | `mckean_vlasov_interacao_eletron.py` | Interação elétron-elétron de campo médio |
| 17 | `derivada_fracionaria_caputo.py`     | Operador de Caputo |
| 18 | `pinn_fracionaria_ruido.py`          | fPINN com expoente de Hurst |
| 19 | `perda_conservacao_probabilidade.py` | Penalização ∫p = 1 |
| 20 | `pinn_multicabecas_estados.py`       | Arquitetura multi-head para N elétrons |
| 21 | `balanceamento_adaptativo_gradientes.py` | Pesos de perda auto-adaptativos |
| 22 | `reamostragem_adaptativa_residuos.py`| Reamostragem nas bordas dos diamantes |
| 23 | `otimizador_hibrido_treinamento.py`  | Adam + L-BFGS |
| 24 | `calculador_gregas_quanticas.py`     | gm, gds, ∂I/∂T |
| 25 | `simulador_monte_carlo_validacao.py` | Gillespie / Monte Carlo cinético |
| 26 | `gerador_diamantes_coulomb.py`       | Mapa I(V_D, V_G) |
| 27 | `pinn_inversa_calibracao.py`         | Calibração inversa de parâmetros |
| 28 | `quantificacao_incerteza_bayesiana.py`| B-PINN / MC-Dropout |
| 29 | `monitoramento_tensorboard.py`       | Logger TensorBoard |
| 30 | `analisador_metricas_convergencia.py`| L2, RMSE, resíduo máximo |
| 31 | `exportador_graficos_convergencia.py`| Visualização de diamantes |
| 32 | `orquestrador_principal_transporte.py`| Pipeline completo |

---

## Instalação

```bash
# Clone ou descompacte o diretório
cd SET_PINN_Framework

# (Recomendado) Crie um ambiente virtual
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate           # Windows

# Instale as dependências
pip install -r requirements.txt
```

**Requisitos mínimos:**
- Python ≥ 3.9
- PyTorch ≥ 2.0 (CPU ou CUDA)
- NumPy, Matplotlib

---

## Execução Rápida

```bash
python main.py
```

O script:
1. Carrega a configuração padrão do SET (temperatura criogênica ~50 mK)
2. Gera pontos de colocalização via Latin Hypercube Sampling
3. Treina a PINN de Fokker-Planck com otimização híbrida (Adam + L-BFGS)
4. Gera o mapa de **Diamantes de Coulomb** (`diamantes_coulomb.png`)
5. Registra perdas no TensorBoard (pasta `runs/set_pinn`)

---

## Física Implementada

### Energia de carregamento
\[
E_C = \frac{e^2}{2 C_\Sigma}, \quad C_\Sigma = C_S + C_D + C_G + C_P
\]

### Variação de energia livre
\[
\Delta F_i^\pm(n,V_D,V_G) = \frac{e}{C_\Sigma}\left[\frac{e}{2}\pm(ne-Q_G)\mp(C_\Sigma-C_i)V_i\pm\sum_{j\neq i}C_j V_j\right]
\]

### Taxas de tunelamento
\[
\Gamma_i^\pm(n)=\frac{1}{e^2 R_T^{(i)}}\frac{-\Delta F_i^\pm}{1-\exp(\Delta F_i^\pm/k_B T_e)}
\]

### Equação mestra
\[
\frac{\partial P(n,t)}{\partial t}=\Gamma^+(n-1)P(n-1)+\Gamma^-(n+1)P(n+1)-\bigl[\Gamma^+(n)+\Gamma^-(n)\bigr]P(n)
\]

### Fokker-Planck contínua
\[
\frac{\partial p}{\partial t}=-\frac{\partial}{\partial q}\bigl[D_1 p\bigr]+\frac12\frac{\partial^2}{\partial q^2}\bigl[D_2 p\bigr]
\]

---

## Uso Modular

Cada módulo pode ser importado isoladamente:

```python
from set_pinn.configuracao_dispositivo import criar_configuracao_padrao
from set_pinn.taxas_tunelamento import taxas_tunelamento
from set_pinn.gerador_diamantes_coulomb import mapa_corrente

cfg = criar_configuracao_padrao()
# ...
```

---

## Licença e Citação

Este código é distribuído para fins acadêmicos e de pesquisa.  
Ao utilizá-lo, cite:

> Wilcke, Luiz Tiago. *Redes Neurais Informadas pela Física – Volume II* (transposição para transporte quântico em SET).

---

**Contato / Autoria:** Luiz Tiago Wilcke  
**Data de geração do pacote:** Agosto 2026
