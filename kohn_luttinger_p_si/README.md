# Teoria da Massa Efetiva com Correção de Célula Central e Acoplamento Órbita-Vale (Kohn–Luttinger)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Átomo de **fósforo substitucional (³¹P)** no silício: elétron confinado em torno do núcleo \(+e\). O confinamento e a **correção de célula central** quebram a degenerescência de **6 vales** da banda de condução nos estados \(1s\):

| Simetria | Degenerescência | \(E_b\) (exp.) |
|----------|-----------------|----------------|
| \(1s(A_1)\) | singlete | \(\approx 45.6\,\mathrm{meV}\) |
| \(1s(T_2)\) | triplete | \(\approx 33.9\,\mathrm{meV}\) |
| \(1s(E)\) | dublete | \(\approx 31.3\,\mathrm{meV}\) |

---

## 1. Equações

### Hamiltoniano EMA + célula central

$$
\left[
-\frac{\hbar^2}{2}
\nabla\cdot
\left(\frac{1}{\mathbf{m}^*}\nabla\right)
+
V_{\mathrm{coul}}(r)
+
V_{\mathrm{cc}}(r)
\right]
\psi_j(r)
=
E_j\psi_j(r)
$$

### Expansão multi-vale

$$
\psi(r)
=
\sum_{\mu=1}^{6}
\alpha_\mu F_\mu(r)\,\phi_\mu(r)
$$

### Potenciais

$$
V_{\mathrm{coul}}(r)
=
-\frac{e^2}{4\pi\varepsilon_r(r)\varepsilon_0 r},
\qquad
V_{\mathrm{cc}}(r)
=
-V_0\exp(-r/r_0)
$$

### Forma radial efetiva (implementação)

$$
-\frac{\hbar^2}{2m^*}
\left(F''+\frac{2}{r}F'\right)
+
V(r)F
=
EF
$$

---

## 2. Resíduo PINN

$$
\mathcal{L}
=
\sum_j
\bigl\lVert\hat H_{\mathrm{EMA}}F_j-E_j F_j\bigr\rVert^2
+
\lambda_{\mathrm{ortho}}
\sum_{j\neq k}
\left\lvert\int F_j F_k\,4\pi r^2 dr\right\rvert^2
+
\lambda_{\mathrm{norm}}
\sum_j
\left\lvert\int\lvert F_j\rvert^2 4\pi r^2 dr-1\right\rvert^2
$$

Canais \(j\in\{A_1,T_2,E\}\) com ordenação \(E_{A_1}<E_{T_2}\le E_E\).

---

## 3. Estrutura

```
kohn_luttinger_p_si/
├── src/
│   ├── fisica_ema.py
│   ├── rede_pinn_ema.py
│   ├── residuo_ema.py
│   └── treinamento_ema.py
├── examples/kohn_luttinger.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/kohn_luttinger.py
```

---

**© 2026 Luiz Tiago Wilcke**
