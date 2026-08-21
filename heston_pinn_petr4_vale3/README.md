# Precificação e Hedging de Ações de Alta Liquidez (PETR4 e VALE3) sob Heston PINN

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

EDP **bidimensional de Heston** com correlação negativa acentuada (\(\rho<0\)) entre retorno e volatilidade — típica de ações de **commodities** na B3 (**PETR4**, **VALE3**). A PINN \(V_{NN}(S,v,t)\) fornece preço contínuo e **gregas** (\(\Delta\), \(\Gamma\), Vanna) para rebalanceamento de mesas de derivativos.

---

## 1. Mercado

| Ativo | Características |
|-------|-----------------|
| **PETR4** | Petróleo; skew negativo forte |
| **VALE3** | Minério; dependência de commodities |
| Opções B3 | Alta liquidez, assimetria de vol |

---

## 2. Formulação Matemática

### Dinâmica de Heston

$$
dS = r S\,dt + \sqrt{v}\,S\,dW^1,
\qquad
dv = \kappa(\theta-v)\,dt + \xi\sqrt{v}\,dW^2
$$

$$
\langle dW^1,dW^2\rangle = \rho\,dt,\quad \rho<0
$$

### EDP de Heston

$$
\frac{\partial V}{\partial t}
+\frac12 v S^2\frac{\partial^2 V}{\partial S^2}
+\rho\xi v S\frac{\partial^2 V}{\partial S\partial v}
+\frac12\xi^2 v\frac{\partial^2 V}{\partial v^2}
+r S\frac{\partial V}{\partial S}
+\kappa(\theta-v)\frac{\partial V}{\partial v}
-r V
=0
$$

### Gregas (via diferenças / Autograd)

$$
\Delta=\frac{\partial V}{\partial S},\quad
\Gamma=\frac{\partial^2 V}{\partial S^2},\quad
\mathrm{Vanna}=\frac{\partial^2 V}{\partial S\partial v}
$$

---

## 3. Estrutura

```
heston_pinn_petr4_vale3/
├── src/
│   ├── heston.py
│   ├── rede_pinn_heston.py   # V + gregas
│   ├── residuo_heston.py
│   └── treinamento_heston.py
├── examples/heston_petr4.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/heston_petr4.py
```

---

**© 2026 Luiz Tiago Wilcke**
