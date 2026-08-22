# Tight-Binding com Flutuação Isotópica Estocástica (Si²⁸ / Si²⁹ / Si³⁰)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

A **desordem de massa isotópica** na rede de silício introduz perturbações estocásticas nas energias de ponto zero que **quebram a degenerescência residual** dos estados excitados tipo \(1s(T_2)\).

---

## 1. Física

| Isótopo | Massa (u) | Abundância natural |
|---------|-----------|--------------------|
| ²⁸Si | 27.977 | ~92.2% |
| ²⁹Si | 28.977 | ~4.7% |
| ³⁰Si | 29.974 | ~3.1% |

\[
\bar M_{\mathrm{Si}} \approx 28.086\,\mathrm{u}
\]

---

## 2. Equações

### Perturbação isotópica onsite

$$
\delta\epsilon_i
=
\alpha_{\mathrm{iso}}
\left(
\frac{M_i-\bar M_{\mathrm{Si}}}{\bar M_{\mathrm{Si}}}
\right),
\qquad
M_i\in\{27.97,\,28.97,\,29.97\}\,\mathrm{u}
$$

### Problema de autovalor TB

$$
\sum_{j,\beta}
H_{i\alpha,j\beta}^{\mathrm{SK}} C_{j\beta}
+
\bigl(\delta\epsilon_i + V_P(R_i)\bigr) C_{i\alpha}
=
E\,C_{i\alpha}
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\sum_\alpha
\left\lVert
\hat H_{\mathrm{TB},\alpha}\mathbf{C}
+
\bigl(\delta\epsilon+V_P\bigr)C_\alpha
-
E C_\alpha
\right\rVert^2
+
\lambda_{\mathrm{norm}}
\bigl\lvert
\lVert\mathbf{C}\rVert^2-1
\bigr\rvert^2
$$

---

## 4. Estrutura

```
tb_isotopico_si/
├── src/
│   ├── fisica_iso.py
│   ├── rede_pinn_iso.py
│   ├── residuo_iso.py
│   └── treinamento_iso.py
├── examples/tb_isotopico.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/tb_isotopico.py
```

---

**© 2026 Luiz Tiago Wilcke**
