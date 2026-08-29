# PINN Margem Equatorial Brasileira

**Autor:** Luiz Tiago Wilcke  
**Baseado em:** *Redes Neurais Informadas pela Física – Volume 3* (Dinâmica Multifásica, Sistemas de Elevação Artificial e Completações Inteligentes)

---

## Descrição

Framework completo de **Redes Neurais Informadas pela Física (PINNs)** adaptado para exploração e modelagem de reservatórios de petróleo na **Margem Equatorial do Brasil** (Foz do Amazonas, Pará-Maranhão, Barreirinhas, Ceará e Potiguar).

Utiliza dados reais públicos da EPE, ANP e Petrobras (volumes *in place* ≈ 17,7 bilhões boe NW Foz do Amazonas, fator de recuperação 35 %, porosidades 15–30 %, permeabilidades típicas de turbiditos, API leve/médio, lâmina d’água ≈ 2 900 m).

Variáveis em português. 25 módulos cobrindo os tópicos do livro adaptados à região.

---

## Estrutura

```
pinn_margem_equatorial/
├── README.md
├── requirements.txt
├── dados_reais.py
├── utilitarios.py
├── main.py
├── modulos/
│   ├── modulo_01_fundamentos_petrofisica.py
│   ├── modulo_02_escoamento_poco_vertical.py
│   ├── ...
│   └── modulo_25_placeholder.py
└── docs/
```

---

## Instalação

```bash
pip install -r requirements.txt
```

## Uso Rápido

```bash
python main.py
python -m modulos.modulo_01_fundamentos_petrofisica
```

---

## Dados Reais Utilizados

| Parâmetro                        | Valor                          |
|----------------------------------|--------------------------------|
| Volume *in place* (NW Foz)       | 17,7 bilhões boe               |
| Volume recuperável               | ≈ 6,2 bilhões boe (RF 35 %)    |
| Porosidade média                 | 0,22 (15–30 %)                 |
| Permeabilidade média             | 150 mD (50–400 mD)             |
| °API médio                       | 30–35                          |
| Lâmina d’água (Morpho)           | 2 886 m                        |
| Formações                        | Limoeiro, Travosas (turbiditos)|

---

## Formulação Matemática do Modelo

Todas as equações abaixo são implementadas como **resíduos físicos** nas funções de perda das PINNs (`L_phys`), calculados via diferenciação automática (Autograd).

### 1. Lei de Darcy (escoamento monofásico)

A velocidade de filtração (fluxo de Darcy) em meio poroso anisotrópico é dada por:

$$
\mathbf{u} = -\frac{\mathbf{K}}{\mu}\bigl(\nabla P - \rho\mathbf{g}\bigr)
$$

onde \(\mathbf{K}\) é o tensor de permeabilidade, \(\mu\) a viscosidade dinâmica, \(P\) a pressão e \(\mathbf{g}\) a aceleração da gravidade.

Em forma escalar unidimensional (direção \(x\)):

$$
u = -\frac{k}{\mu}\frac{\partial P}{\partial x}
$$

### 2. Equação Geral da Difusividade Monofásica

Acoplando a Lei de Darcy com a conservação de massa:

$$
\nabla\cdot\Bigl(\frac{\mathbf{K}}{\mu}\nabla P\Bigr) = \phi c_t\frac{\partial P}{\partial t} + \frac{q}{\rho_{\mathrm{std}}}
$$

onde \(\phi\) é a porosidade, \(c_t = c_l + c_f\) a compressibilidade total (fluido + formação) e \(q\) a vazão de fonte/sumidouro.

Para meio isotrópico homogêneo 2-D:

$$
\frac{k}{\mu}\left(\frac{\partial^2 P}{\partial x^2} + \frac{\partial^2 P}{\partial y^2}\right) = \phi c_t\frac{\partial P}{\partial t}
$$

### 3. Escoamento Radial Estacionário (Equação de Dupuit)

Em coordenadas cilíndricas, regime permanente (\(\partial P/\partial t = 0\)):

$$
\frac{1}{r}\frac{d}{dr}\left(r\frac{dP}{dr}\right) = 0
$$

Solução analítica com condições de contorno \(P(r_w)=P_{wf}\) e \(P(r_e)=P_e\):

$$
P(r) = P_{wf} + \frac{P_e - P_{wf}}{\ln(r_e/r_w)}\ln\left(\frac{r}{r_w}\right)
$$

Vazão de produção:

$$
Q = \frac{2\pi k h (P_e - P_{wf})}{\mu\ln(r_e/r_w)}
$$

### 4. Dinâmica Multifásica – Buckley-Leverett

Fração de fluxo de água:

$$
f_w(S_w) = \frac{1}{1 + \dfrac{k_{ro}(S_w)\mu_w}{k_{rw}(S_w)\mu_o}}
$$

Equação de transporte hiperbólica:

$$
\frac{\partial S_w}{\partial t} + \frac{u_t}{\phi}\frac{df_w}{dS_w}\frac{\partial S_w}{\partial x} = 0
$$

**Histerese (Modelo de Killough + Land):**

$$
S_{gt} = \frac{S_{g,hy}}{1 + C_{\mathrm{Land}}S_{g,hy}},\qquad
C_{\mathrm{Land}} = \frac{1}{S_{gr,\max}} - \frac{1}{S_{g,\max}^{dr}}
$$

$$
k_{rg}^{imb}(S_g) = k_{rg}^{dr}(S_{g,hy})\left(\frac{S_g - S_{gt}}{S_{g,hy} - S_{gt}}\right)^\alpha
$$

A PINN incorpora memória histórica \(S_{g,\max}^{hist}(x,t) = \max_{\tau\in[0,t]}S_g(x,\tau)\).

### 5. Gradiente de Pressão Vertical no Poço

$$
\frac{dP}{dz} = \underbrace{\rho_m g\sin\theta}_{\text{gravitacional}} + \underbrace{\frac{f\rho_m v^2}{2D}}_{\text{friccional}} + \underbrace{\rho_m v\frac{dv}{dz}}_{\text{acelerativo}}
$$

onde \(\rho_m\) é a densidade da mistura multifásica, \(f\) o fator de atrito e \(D\) o diâmetro interno.

### 6. Equação de Estado de Peng-Robinson (Termodinâmica Composicional)

$$
P = \frac{RT}{V_m - b} - \frac{a(T)}{V_m(V_m + b) + b(V_m - b)}
$$

com parâmetros:

$$
a_i(T) = \Omega_a\frac{R^2T_{c,i}^2}{P_{c,i}}\alpha_i(T_{r,i},\omega_i),\qquad
b_i = \Omega_b\frac{RT_{c,i}}{P_{c,i}}
$$

$$
\alpha_i = \bigl[1 + m_i\bigl(1 - \sqrt{T_{r,i}}\bigr)\bigr]^2,\qquad
m_i = 0{,}37464 + 1{,}54226\omega_i - 0{,}26992\omega_i^2
$$

### 7. Função de Perda Multiobjetivo da PINN

A rede neural \(u_\theta(x,t)\) é treinada minimizando:

$$
\mathcal{L}(\boldsymbol{\theta}) = \lambda_{\mathrm{data}}\mathcal{L}_{\mathrm{data}} + \lambda_{\mathrm{phys}}\mathcal{L}_{\mathrm{phys}} + \lambda_{\mathrm{bc}}\mathcal{L}_{\mathrm{bc}} + \lambda_{\mathrm{ic}}\mathcal{L}_{\mathrm{ic}}
$$

onde:

$$
\mathcal{L}_{\mathrm{phys}} = \frac{1}{N_{\mathrm{col}}}\sum_{i=1}^{N_{\mathrm{col}}}\bigl|R\bigl[u_\theta\bigr](x_i,t_i)\bigr|^2
$$

e \(R[\cdot]\) é o residual da equação diferencial parcial (calculado por Autograd).

Equivalência estatística: estimativa de **Máxima A Posteriori (MAP)**.

### 8. Regularização de Variação Total (inversão de permeabilidade)

$$
\mathcal{L}_{\mathrm{TV}} = \int_\Omega\bigl|\nabla\log k(\mathbf{x})\bigr|\,d\mathbf{x}
$$

Preserva descontinuidades geológicas (falhas, lentes de turbiditos).

### 9. Gas Lift – Curva de Performance

Relação empírica embutida na PINN de elevação artificial:

$$
q_o(q_g,z_{\mathrm{inj}}) \approx a\cdot\frac{q_g}{b + q_g}\cdot\bigl(1 - c\cdot z_{\mathrm{inj}}/z_{\max}\bigr)
$$

O ponto ótimo de injeção de gás é obtido por maximização diferenciável via gradiente.

---

## Módulos Implementados

| Módulo | Conteúdo principal                                      | Status      |
|--------|---------------------------------------------------------|-------------|
| 01     | Darcy radial + difusividade monofásica                  | Completo    |
| 02     | Gradiente de pressão vertical multifásico               | Completo    |
| 03     | Formulação PINN, perda multiobjetivo, MAP               | Completo    |
| 04     | Inversão de permeabilidade + regularização TV           | Completo    |
| 05     | Gas Lift / BCS – otimização física                      | Completo    |
| 06–25  | Estruturas prontas (Buckley-Leverett, XPINN, FNO, …)   | Expandíveis |

---

## Autor

**Luiz Tiago Wilcke**  
Bacharel em Estatística  
Especialista em Deep Learning Científico e Engenharia de Petróleo  

*VOLUME III: COMPUTAÇÃO NEURAL APLICADA À SUBSUPERFÍCIE – Adaptação Margem Equatorial Brasileira*
