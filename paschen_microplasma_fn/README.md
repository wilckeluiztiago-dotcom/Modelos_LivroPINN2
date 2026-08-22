# Ruptura Eletrostática por Efeito de Proximidade e Microplasma Nanométrico (Lei de Paschen Modificada)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Em gaps **sub-5 nm** entre nanofios, a quebra dielétrica **não segue** a curva clássica de Paschen. A **emissão de elétrons por campo** (Fowler–Nordheim) substitui a ionização volumétrica de Townsend.

---

## 1. Física

| Regime | Mecanismo | Escala |
|--------|-----------|--------|
| Paschen clássico | Avalanche de Townsend | \(d \gtrsim 5\,\mu\mathrm{m}\) |
| **FN / microplasma nm** | Emissão de campo quântico | \(d \lesssim 5\,\mathrm{nm}\) |

---

## 2. Equações

### Poisson

$$
\nabla\cdot(\varepsilon\nabla\phi)=-q(n_i-n_e)
$$

### Continuidade eletrônica

$$
\frac{\partial n_e}{\partial t}
+
\nabla\cdot(\mathbf{v}_e n_e-D_e\nabla n_e)
=
\alpha\lvert\mathbf{v}_e\rvert n_e
+
G_{\mathrm{FN}}(\mathcal{E})
$$

### Fonte Fowler–Nordheim (cátodo)

$$
G_{\mathrm{FN}}(\mathcal{E})
=
A_{\mathrm{FN}}\,\mathcal{E}^2
\exp\!\left(-\frac{B_{\mathrm{FN}}}{\mathcal{E}}\right)
\delta(x-x_{\mathrm{catodo}})
$$

com \(\mathcal{E}=\lvert\nabla\phi\rvert\), \(\mathbf{v}_e=-\mu_e\nabla\phi\).

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\bigl\lVert
\nabla\cdot(\varepsilon\nabla\phi)+q(n_i-n_e)
\bigr\rVert^2
+
\bigl\lVert
\partial_t n_e
+
\nabla\cdot(\mu_e\nabla\phi\,n_e-D_e\nabla n_e)
-
\alpha\lvert\mu_e\nabla\phi\rvert n_e
-
G_{\mathrm{FN}}(\nabla\phi)
\bigr\rVert^2
$$

---

## 4. Estrutura

```
paschen_microplasma_fn/
├── src/
│   ├── fisica_paschen.py
│   ├── rede_pinn_fn.py
│   ├── residuo_fn.py
│   └── treinamento_fn.py
├── examples/paschen_fn.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/paschen_fn.py
```

---

**© 2026 Luiz Tiago Wilcke**
