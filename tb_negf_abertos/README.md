# Tight-Binding Acoplado a Eletrodos Abertos (Formalismo Atomístico TB-NEGF)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Conexão do átomo de **fósforo** a reservatórios semi-infinitos de silício degeneradamente dopado (**leads \(n^+\)**) via matrizes de auto-energia de contato \(\Sigma_L(E)\) e \(\Sigma_R(E)\) no nível atomístico.

---

## 1. Física

| Elemento | Papel |
|----------|--------|
| Canal TB | Cadeia Si com \(V_P\) no centro (\(^{31}\)P) |
| Leads \(n^+\) | Reservatórios semi-infinitos |
| \(\Sigma_{L,R}(E)\) | Auto-energias de contato |
| \(G^R(E)\) | Função de Green retardada |
| \(\mathcal{T}(E)\) | Transmissão Landauer |

---

## 2. Equações

### Equação de Dyson / Green retardada

$$
\bigl[
E\,\mathbb{I}
-
H_{\mathrm{TB}}
-
V_P
-
\Sigma_L(E)
-
\Sigma_R(E)
\bigr]
G^R(E)
=
\mathbb{I}
$$

### Larguras de nível

$$
\Gamma_{L,R}
=
i\bigl(\Sigma_{L,R}-\Sigma_{L,R}^\dagger\bigr)
$$

### Transmissão

$$
\mathcal{T}(E)
=
\mathrm{Tr}
\bigl[
\Gamma_L(E)\,G^R(E)\,\Gamma_R(E)\,G^{R\dagger}(E)
\bigr]
$$

### Corrente Landauer

$$
I_{ds}
=
\frac{2e}{h}
\int
\mathcal{T}(E)\,[f_L(E)-f_R(E)]\,dE
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\left\lVert
\bigl(E\mathbb{I}-\hat H_{\mathrm{TB}}-V_P-\Sigma_L-\Sigma_R\bigr)
\mathbf{G}^R_\theta(E)
-
\mathbb{I}
\right\rVert^2
+
\lambda_T
\left\lVert
I_{ds}
-
\frac{2e}{h}
\int\mathcal{T}_\theta(E)[f_L-f_R]\,dE
\right\rVert^2
$$

A rede \(f_\theta(E)\) produz \(\mathrm{Re}\,G^R\) e \(\mathrm{Im}\,G^R\).

---

## 4. Estrutura

```
tb_negf_abertos/
├── src/
│   ├── fisica_negf.py       # H, Σ, G^R, T(E), I
│   ├── rede_pinn_negf.py
│   ├── residuo_negf.py
│   └── treinamento_negf.py
├── examples/tb_negf.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/tb_negf.py
```

---

**© 2026 Luiz Tiago Wilcke**
