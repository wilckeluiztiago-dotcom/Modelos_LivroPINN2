# Quantização de Condutância de Landauer–Büttiker em Constrições e Junções 1D

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Quando o comprimento do nanofio é menor que o livre caminho médio inelástico (\(L < \ell_{\mathrm{in}}\)), o transporte é **balístico** e a condutância quantiza em múltiplos de

$$
G_0 = \frac{2e^2}{h}.
$$

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| **Sub-bandas** | Modos transversais \(\psi_n(y)\) com energias \(E_n\) |
| **Transmissão** | \(\mathcal{T}_n(E)\approx 1\) se \(E>E_n\) (balístico ideal) |
| **Landauer** | Cada modo aberto contribui \(G_0\) à condutância |
| **Constrição** | Poço de confinamento transversal \(V_{\mathrm{conf}}(y)\) |

---

## 2. Equações

### Schrödinger transversal

$$
-\frac{\hbar^2}{2m^*}\nabla^2\psi_n(x,y)
+
\bigl[V(x,y)+V_{\mathrm{conf}}(y)\bigr]\psi_n
=
E_n\psi_n
$$

Em constrição uniforme (modos 1D em \(y\)):

$$
-\frac{\hbar^2}{2m^*}\frac{d^2\psi_n}{dy^2}
+
V_{\mathrm{conf}}(y)\,\psi_n
=
E_n\psi_n
$$

### Corrente Landauer–Büttiker

$$
I_{ds}
=
\frac{2e}{h}
\sum_{n=1}^{M(E)}
\int
\mathcal{T}_n(E)
\bigl[f_S(E)-f_D(E)\bigr]
\,dE
$$

### Condutância quantizada (T = 0, balístico)

$$
G = G_0 \times (\text{número de modos com } E_n < \mu)
$$

---

## 3. Resíduo PINN

A rede resolve os autoestados transversais:

$$
\mathcal{L}
=
\sum_n
\bigl\lVert
\hat H_{1D}\psi_n - E_n\psi_n
\bigr\rVert^2
+
\lambda_{\mathrm{ortho}}
\sum_{n,m}
\Bigl(
\int\psi_n^*\psi_m\,dy - \delta_{nm}
\Bigr)^2
+
\lambda_{\mathrm{BC}}
\sum_n
\bigl(
\psi_n(0)^2+\psi_n(W)^2
\bigr)
$$

com

$$
\hat H_{1D}
=
-\frac{\hbar^2}{2m^*}\partial_{yy}
+
V_{\mathrm{conf}}(y).
$$

---

## 4. Estrutura

```
landauer_buttiker_condutancia/
├── src/
│   ├── fisica_landauer.py     # modos, I(V), G/G0
│   ├── rede_pinn_modos.py
│   ├── residuo_modos.py
│   └── treinamento_modos.py
├── examples/landauer_buttiker.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/landauer_buttiker.py
```

---

**© 2026 Luiz Tiago Wilcke**
