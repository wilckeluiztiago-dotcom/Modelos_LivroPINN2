# Modelo Tight-Binding Atomístico Multi-Orbital (\(sp^3d^5s^*\)) com Potencial de Coulomb Triado

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Representação da estrutura eletrônica **átomo a átomo** sobre a rede de diamante do silício, com **10 orbitais por átomo** e o potencial microscópico central do fósforo.

---

## 1. Física

| Elemento | Descrição |
|----------|-----------|
| Base | \(s,p_x,p_y,p_z,d_{xy},d_{yz},d_{zx},d_{x^2-y^2},d_{z^2},s^*\) |
| Rede | Diamante (Si) |
| Doador | \(^{31}\)P substitucional + \(V_P\) triado |
| \(U_{cc}\) | Correção de célula central no sítio P |

---

## 2. Equações

### Problema de autovalor TB

$$
\sum_{j,\beta}
H_{i\alpha,\,j\beta}\,C_{j\beta}
+
V_P(R_i)\,C_{i\alpha}
=
E\,C_{i\alpha}
$$

### Potencial de Coulomb triado + célula central

$$
V_P(R_i)
=
-\frac{e^2}{4\pi\varepsilon_0\varepsilon_r(R_i)\lvert R_i-R_P\rvert}
\left(1-e^{-\lvert R_i-R_P\rvert/r_{\mathrm{core}}}\right)
+
U_{cc}\,\delta_{i,P}
$$

---

## 3. Resíduo PINN

A rede prevê coeficientes orbitais contínuos \(C_\alpha(r)\):

$$
\mathcal{L}
=
\sum_{\alpha=1}^{10}
\bigl\lVert
\hat H_{\mathrm{TB},\alpha}\mathbf{C}(r)+V_P(r)C_\alpha(r)-E C_\alpha(r)
\bigr\rVert^2
+
\lambda_{\mathrm{norm}}
\left\lvert
\sum_\alpha\int\lvert C_\alpha\rvert^2\,d^3r-1
\right\rvert^2
$$

Na prática, o resíduo é avaliado nos **sítios atômicos** do cluster com \(H\) montado explicitamente.

---

## 4. Estrutura

```
tight_binding_sp3d5s/
├── src/
│   ├── fisica_tb.py          # cluster, H, V_P, diagonalização
│   ├── rede_pinn_tb.py
│   ├── residuo_tb.py
│   └── treinamento_tb.py
├── examples/tb_sp3d5s.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/tb_sp3d5s.py
```

---

**© 2026 Luiz Tiago Wilcke**
