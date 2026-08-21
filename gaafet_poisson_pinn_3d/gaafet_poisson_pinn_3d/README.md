# PINN 3D Livre de Malhas para Equação de Poisson em GAAFETs

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Ano:** 2026  

Base: *Redes Neurais Informadas pela Física — Aplicações no Mercado Financeiro* (Volume II)  
e formalismo geral de PINNs (Caps. 2–3: aproximação universal, perda composta, LHS, otimização híbrida).

---

## 1. Visão Geral

Mapeamento do **potencial eletrostático contínuo** em nanoestruturas **Gate-All-Around (GAAFET)** pela equação de Poisson

$$
\nabla \cdot \bigl(\varepsilon(x,y,z)\,\nabla\phi\bigr) = -\rho(x,y,z)
$$

com uma **PINN 3D livre de malhas**: sem discretização espacial rígida (FDM/FEM), eliminando a maldição da dimensionalidade típica de grades 3D.

---

## 2. Fenômeno Físico

Em GAAFETs o gate envolve completamente um canal nanowire:

- Canal de Si (raio $R_{\mathrm{canal}}$) com permitividade $\varepsilon_{\mathrm{Si}}$
- Óxido (casca $R_{\mathrm{canal}}<r\le R_{\mathrm{ox}}$) com $\varepsilon_{\mathrm{ox}}$
- Gate metálico em $r=R_{\mathrm{ox}}$ com potencial $V_G$ fixo
- Source / Drain nas faces $x=0$ e $x=L$

A solução $\phi(x,y,z)$ determina o campo elétrico e a densidade de carreadores no regime nanoescala.

---

## 3. Equações e PINN

### Poisson 3D

$$
\nabla\cdot(\varepsilon\nabla\phi)=-\rho
\quad\Leftrightarrow\quad
\varepsilon\Delta\phi+\nabla\varepsilon\cdot\nabla\phi+\rho=0
$$

### Rede

$$
\phi_\theta:\mathbb{R}^3\to\mathbb{R},\qquad
\theta=\{\text{pesos e vieses do MLP}\}
$$

### Perda composta (Cap. 2.5)

$$
\mathcal{J}(\theta)
=\underbrace{\frac1{N_c}\sum_{i=1}^{N_c}\bigl|\nabla\cdot(\varepsilon\nabla\phi_\theta)+\rho\bigr|^2_{X_i}}_{\text{resíduo PDE}}
+\lambda_{\mathrm{BC}}\underbrace{\frac1{N_b}\sum_{j=1}^{N_b}\bigl|\phi_\theta-g\bigr|^2_{\partial\Omega_j}}_{\text{contorno}}
$$

### Amostragem

Pontos de colocation via **Latin Hypercube Sampling** (Cap. 3.5) — livre de malha.

---

## 4. Estrutura

```
gaafet_poisson_pinn_3d/
├── src/
│   ├── rede_pinn3d.py          # MLP 3D (Cap. 2)
│   ├── geometria_gaafet.py     # Nanowire + óxido + gate
│   ├── residuo_poisson.py      # Perda composta
│   ├── treinamento.py          # Otimização (Cap. 3.6)
│   └── utils.py                # LHS
├── examples/gaafet_poisson_pinn.py
├── figures/
├── artigo/                     # LaTeX + PDF
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/gaafet_poisson_pinn.py
```

---

## 6. Referência

Wilcke, L. T. *Redes Neurais Informadas pela Física — Aplicações no Mercado Financeiro*, Volume II. Caps. 2–3.

**© 2026 Luiz Tiago Wilcke**
