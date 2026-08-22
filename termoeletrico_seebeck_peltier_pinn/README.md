# Transporte Termoelétrico Não-Linear (Seebeck e Peltier em Escala Atômica)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Geração e absorção localizada de calor nos nós de injeção **fonte/dreno**, com acoplamento cruzado entre conservação de carga e energia (termo de Thomson).

---

## 1. Física

| Efeito | Descrição |
|--------|-----------|
| **Seebeck** | Gradiente de \(T\) induz corrente / FEM |
| **Peltier** | Corrente transporta calor (\(\Pi = ST\)) |
| **Thomson** | Aquecimento/resfriamento ao longo do gradiente |

---

## 2. Equações

### Corrente elétrica (Ohm + Seebeck)

$$
\mathbf{J}
=
-\sigma\nabla\phi
-\sigma S\nabla T
$$

### Coeficiente de Peltier

$$
\Pi = S\,T
$$

### Fluxo de calor

$$
\mathbf{q}
=
\Pi\,\mathbf{J}
-\kappa\nabla T
$$

### Conservação de energia

$$
\nabla\cdot\mathbf{q}
=
\mathbf{J}\cdot\mathcal{E}
-
\nabla\cdot(\Pi\mathbf{J})
$$

com \(\mathcal{E}=-\nabla\phi\).

### Conservação de carga

$$
\nabla\cdot\mathbf{J}=0
$$

### Forma 1D (implementação)

$$
J
=
-\sigma\,\partial_x\phi
-\sigma S\,\partial_x T
$$

$$
\partial_x J = 0
$$

$$
\partial_x q
=
J\,(-\partial_x\phi)
-
\partial_x(\Pi J)
$$

---

## 3. Resíduo PINN (PyTorch + Autograd)

$$
\mathcal{L}
=
\bigl\lVert\partial_x J\bigr\rVert^2
+
\bigl\lVert
\partial_x q
-
J\cdot\mathcal{E}
+
\partial_x(\Pi J)
\bigr\rVert^2
+
\lambda_{\mathrm{BC}}
\bigl(
\lVert\phi-\phi_{\mathrm{BC}}\rVert^2
+
\lVert T-T_{\mathrm{BC}}\rVert^2
\bigr)
$$

A rede \(f_\theta(x)=(\phi(x),\,T(x))\) é treinada com **autograd** para todas as derivadas.

---

## 4. Estrutura

```
termoeletrico_seebeck_peltier_pinn/
├── src/
│   ├── fisica_termo.py
│   ├── rede_pinn_termo.py      # PyTorch nn.Module
│   ├── residuo_termo.py        # autograd
│   └── treinamento_termo.py    # Adam
├── examples/termo_seebeck_peltier.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/termo_seebeck_peltier.py
```

---

**© 2026 Luiz Tiago Wilcke**
