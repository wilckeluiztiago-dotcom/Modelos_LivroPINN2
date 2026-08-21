# DGM 6D para Equação de Wigner–Boltzmann em Nanofolhas de 1.6 nm

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Ano:** 2026  

Resolução **livre de malhas** da função de distribuição de quase-probabilidade quântica \(f_W(\mathbf{x},\mathbf{k},t)\) no espaço de fases **6D** (3D espacial + 3D momentos), usando células neurais recorrentes do **Deep Galerkin Method (DGM)** para capturar transporte **quase-balístico** em nanofolhas de espessura \(\approx 1{,}6\,\mathrm{nm}\).

---

## 1. Fenômeno Físico

Em nanofolhas ultratinas (\(L_z \sim 1{,}6\,\mathrm{nm}\)):

1. O transporte é **quase-balístico**: o livre caminho médio torna-se comparável ao comprimento do canal.
2. Efeitos quânticos de interferência e tunelamento exigem uma descrição no **espaço de fases**, não apenas em posição.
3. A função de Wigner \(f_W\) é a análoga quântica da distribuição de Boltzmann clássica; pode assumir valores negativos (interferência).
4. A equação de Wigner–Boltzmann acopla deriva no espaço \(\mathbf{x}\), força no espaço \(\mathbf{k}\) e um termo não-local de potencial (operador de Wigner).

A abordagem DGM evita a **maldição da dimensionalidade** de grades 6D (impraticáveis com FDM/FEM).

---

## 2. Equação de Wigner–Boltzmann (forma completa 6D)

### Função de Wigner

$$
f_W(\mathbf{x},\mathbf{k},t)
=\frac{1}{(2\pi)^3}
\int_{\mathbb{R}^3}
\rho\!\left(\mathbf{x}+\frac{\mathbf{y}}{2},\,\mathbf{x}-\frac{\mathbf{y}}{2},\,t\right)
e^{-i\mathbf{k}\cdot\mathbf{y}}\,d\mathbf{y}
$$

onde \(\rho\) é a matriz densidade de uma partícula.

### Equação de evolução

$$
\frac{\partial f_W}{\partial t}
+\frac{\hbar\mathbf{k}}{m^*}\cdot\nabla_{\mathbf{x}} f_W
+\Theta[V]f_W
= C[f_W]
$$

### Operador de potencial de Wigner (não-local)

$$
\Theta[V]f_W
=\frac{1}{(2\pi)^3\hbar}
\int_{\mathbb{R}^3}\!d\mathbf{k}'
\int_{\mathbb{R}^3}\!d\mathbf{y}\;
\Bigl[V\!\left(\mathbf{x}+\tfrac{\mathbf{y}}{2}\right)
-V\!\left(\mathbf{x}-\tfrac{\mathbf{y}}{2}\right)\Bigr]
e^{-i(\mathbf{k}-\mathbf{k}')\cdot\mathbf{y}}
\,f_W(\mathbf{x},\mathbf{k}',t)
$$

### Limite semiclassico (campo suave)

Quando \(V\) varia lentamente na escala de \(\hbar\),

$$
\Theta[V]f_W \;\longrightarrow\; -\nabla_{\mathbf{x}}V\cdot\nabla_{\mathbf{k}} f_W
$$

recuperando a equação de Boltzmann clássica:

$$
\frac{\partial f}{\partial t}
+\mathbf{v}\cdot\nabla_{\mathbf{x}} f
+\mathbf{F}\cdot\nabla_{\mathbf{k}} f
= C[f],
\qquad
\mathbf{v}=\frac{\hbar\mathbf{k}}{m^*},
\quad
\mathbf{F}=-\nabla V.
$$

### Colisões (BGK / relaxação)

$$
C[f] = -\gamma\,(f - f_{\mathrm{eq}})
$$

com taxa de espalhamento \(\gamma\) (fônons, rugosidade de interface na nanofolha).

---

## 3. Deep Galerkin Method (DGM)

### Arquitetura da célula recorrente

Entrada \(\mathbf{u}=(\mathbf{x},\mathbf{k},t)\in\mathbb{R}^{7}\) (6D + tempo). Estado oculto \(S\in\mathbb{R}^{D}\).

$$
\begin{aligned}
Z &= \sigma(W_z[\mathbf{u},S]+b_z),\\
G &= \sigma(W_g[\mathbf{u},S]+b_g),\\
R &= \sigma(W_r[\mathbf{u},S]+b_r),\\
H &= \tanh(W_h[\mathbf{u},R\odot S]+b_h),\\
S_{\mathrm{new}} &= (1-G)\odot H + Z\odot S.
\end{aligned}
$$

Camada de entrada \(\to\) \(L\) células DGM \(\to\) saída escalar \(f_\theta(\mathbf{u})\).

### Perda composta (livre de malhas)

$$
\mathcal{J}(\theta)
=
\frac{1}{N_c}\sum_{i=1}^{N_c}
\Bigl|
\partial_t f_\theta
+\mathbf{v}\cdot\nabla_{\mathbf{x}} f_\theta
+\mathbf{F}\cdot\nabla_{\mathbf{k}} f_\theta
-C[f_\theta]
\Bigr|^2_{\mathbf{u}_i}
+
\lambda_{\mathrm{IC}}
\frac{1}{N_0}\sum_{j=1}^{N_0}
\bigl|f_\theta(\mathbf{x}_j,\mathbf{k}_j,0)-f_0\bigr|^2
$$

Pontos \(\mathbf{u}_i\) amostrados por **Latin Hypercube** no hipercubo 6D+tempo — **sem malha espacial**.

### Demonstração numérica

A implementação de referência opera na **redução operacional** \((x,k_x,t)\) (transporte 1D efetivo com confinamento em \(z\)), preservando a estrutura DGM extensível a 6D completo:

$$
\partial_t f + \frac{\hbar k_x}{m}\partial_x f + F(x)\partial_{k_x} f = -\gamma f.
$$

---

## 4. Nanofolha de 1.6 nm

| Parâmetro | Valor típico |
|-----------|--------------|
| Espessura \(L_z\) | \(1{,}6\,\mathrm{nm}\) |
| Transporte | plano \(xy\), confinamento em \(z\) |
| Regime | quase-balístico (\(\gamma\) pequeno) |
| Potencial | barreira suave / heterointerface |

---

## 5. Estrutura do Projeto

```
dgm_wigner_boltzmann_6d/
├── src/
│   ├── celula_dgm.py           # Células recorrentes DGM
│   ├── wigner_boltzmann.py     # Física Wigner + nanofolha
│   ├── residuo_wigner.py       # Resíduo e perda composta
│   ├── treinamento.py
│   └── utils.py                # LHS
├── examples/dgm_wigner_nanofolha.py
├── figures/
├── docs/mathematical_model.md
├── artigo/                     # LaTeX + PDF
└── README.md
```

---

## 6. Instalação e Uso

```bash
cd dgm_wigner_boltzmann_6d
pip install -r requirements.txt
python examples/dgm_wigner_nanofolha.py
```

---

## 7. Classes principais (português)

| Nome | Papel |
|------|--------|
| `RedeDGM` / `CelulaDGM` | Rede com células recorrentes |
| `NanofolhaWigner` | Geometria e potencial 1.6 nm |
| `residuo_wigner_reduzido` | Resíduo da EDP de fase |
| `treinar_dgm` | Loop de otimização |
| `amostragem_lhs` | Colocation livre de malha |

---

## 8. Referências conceituais

- Sirignano & Spiliopoulos — Deep Galerkin Method  
- Equação de Wigner–Boltzmann (transporte quântico)  
- Luiz Tiago Wilcke — formalismo de redes informadas por física / EDPs de alta dimensão  

**© 2026 Luiz Tiago Wilcke**
