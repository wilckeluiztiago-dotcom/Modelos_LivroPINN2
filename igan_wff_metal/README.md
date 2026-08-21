# I-GANs para Síntese de Flutuação de Trabalho de Extração de Metal (WFF)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo:** 44  

Jogo **minimax adversário** regularizado por **operadores diferenciais eletrostáticos** (Poisson), para gerar mapas sintéticos de **orientação de grãos metálicos** (TiN/TaN) e **work function fluctuation (WFF)** em nós de **1.6 nm**.

---

## 1. Fenômeno Físico

1. Portas metálicas TiN/TaN formam **grãos** com orientações (100)/(110)/(111) e work functions distintos.
2. Em 1.6 nm, poucos grãos cobrem a nanofolha → **WFF** impacta \(V_t\) e corrente.
3. Dados experimentais de mapas de grãos são escassos → **síntese** via I-GAN.
4. A perda física força \(\phi_G\) (potencial do mapa gerado) a satisfazer **Poisson**.

---

## 2. Equações

### Jogo minimax com regularização física

$$
\min_G \max_D
\;
\mathcal{L}_{\mathrm{GAN}}(G,D)
+
\lambda_{\mathrm{phys}}
\bigl\|
\nabla\cdot(\varepsilon\nabla\phi_G)+\rho
\bigr\|^2
$$

### Perda adversária

$$
\mathcal{L}_{\mathrm{GAN}}
=
\mathbb{E}_{x\sim p_{\mathrm{data}}}\bigl[\log D(x)\bigr]
+
\mathbb{E}_{z\sim p_z}\bigl[\log\bigl(1-D(G(z))\bigr)\bigr]
$$

### Potencial a partir do WF gerado

$$
\phi_G(x)
=
V_S + (V_D-V_S)\frac{x}{L}
+
\alpha\bigl(\overline{\mathrm{WF}}_G(x)-\mathrm{WF}_{\mathrm{ref}}\bigr)
$$

### Resíduo de Poisson

$$
\mathcal{R}_{\mathrm{phys}}
=
\varepsilon\,\partial_{xx}\phi_G + \rho
$$

---

## 3. Arquitetura

| Rede | Entrada | Saída |
|------|---------|--------|
| Gerador \(G\) | \(z\sim\mathcal{N}(0,I)\) | mapa WF \((n_x\times n_y)\) |
| Discriminador \(D\) | mapa WF | \(P(\mathrm{real})\) |

---

## 4. Estrutura

```
igan_wff_metal/
├── src/
│   ├── graos_wff.py       # Voronoi de grãos + WF + Poisson
│   ├── rede_gan.py        # G e D
│   ├── perda_igan.py      # L_GAN + L_phys
│   └── treinamento_igan.py
├── examples/igan_wff.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/igan_wff.py
```

---

**© 2026 Luiz Tiago Wilcke**
