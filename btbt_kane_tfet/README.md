# Tunelamento Quântico Interbandas (BTBT — Kane/Keldysh para TFETs)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Em **TFETs**, elétrons da banda de valência da fonte tunelam através do bandgap para a banda de condução do canal, permitindo inclinação sub-limiar **\(SS < 60\,\mathrm{mV/dec}\)** (abaixo do limite termiônico).

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| BTBT | Band-to-band tunneling |
| Kane/Keldysh | Taxa \(G(\mathcal{E})\) exponencial no campo |
| TFET | Fonte p+ / canal i / dreno n+ |
| SS | Subthreshold swing |

---

## 2. Equações

### Geração Kane

$$
G_{\mathrm{BTBT}}(\mathcal{E})
=
A_{\mathrm{Kane}}
\frac{\mathcal{E}^2}{\sqrt{E_g}}
\exp\!\left(
-B_{\mathrm{Kane}}\frac{E_g^{3/2}}{\mathcal{E}}
\right),
\qquad
\mathcal{E}=\lvert\nabla\phi\rvert
$$

### Continuidade

$$
\nabla\cdot\mathbf{J}_n = q G_{\mathrm{BTBT}}-q R(n,p)
$$

$$
\nabla\cdot\mathbf{J}_p = -q G_{\mathrm{BTBT}}+q R(n,p)
$$

### Poisson

$$
\nabla\cdot(\varepsilon\nabla\phi)=-q(p-n+N_D^+-N_A^-)
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\bigl\lVert\nabla\cdot\mathbf{J}_n-qG+qR\bigr\rVert^2
+
\bigl\lVert\nabla\cdot\mathbf{J}_p+qG-qR\bigr\rVert^2
+
\bigl\lVert\nabla\cdot(\varepsilon\nabla\phi)+q(p-n+N_{\mathrm{net}})\bigr\rVert^2
$$

---

## 4. Estrutura

```
btbt_kane_tfet/
├── src/
│   ├── fisica_btbt.py
│   ├── rede_pinn_btbt.py
│   ├── residuo_btbt.py
│   └── treinamento_btbt.py
├── examples/btbt_tfet.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/btbt_tfet.py
```

---

**© 2026 Luiz Tiago Wilcke**
