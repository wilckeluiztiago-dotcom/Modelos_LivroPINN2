# Modelo Hidrodinâmico de Transporte de Portadores Quentes (Baccarani–Wordeman / Bløtekjaer)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Transporte em semicondutores **sub-10 nm** sob gradientes de campo que causam **velocity overshoot**, com temperatura eletrônica \(T_n(x)\) desacoplada da temperatura da rede \(T_L\).

---

## 1. Física

| Grandeza | Papel |
|----------|--------|
| \(n, \mathbf{v}_n\) | Densidade e velocidade de deriva |
| \(T_n\) | Temperatura dos elétrons (hot carriers) |
| \(\tau_p(T_n)\) | Relaxação de momento |
| \(\tau_w(T_n)\) | Relaxação de energia |
| Overshoot | \(v_n > v_{\mathrm{sat}}\) localmente |

---

## 2. Equações

### Continuidade

$$
\frac{\partial n}{\partial t}+\nabla\cdot(n\mathbf{v}_n)=0
$$

### Momento

$$
\frac{\partial(n\mathbf{v}_n)}{\partial t}
+
\nabla\cdot(n\mathbf{v}_n\otimes\mathbf{v}_n)
=
-\frac{qn}{m^*}\mathbf{E}
-
\frac{1}{m^*}\nabla(n k_B T_n)
-
\frac{n\mathbf{v}_n}{\tau_p(T_n)}
$$

### Energia

$$
\frac{\partial(n W_n)}{\partial t}
+
\nabla\cdot(\mathbf{v}_n n W_n)
=
-q n\mathbf{v}_n\cdot\mathbf{E}
-
\nabla\cdot(n k_B T_n\mathbf{v}_n+\mathbf{Q}_n)
-
n\frac{W_n-W_0}{\tau_w(T_n)}
$$

$$
W_n=\frac32 k_B T_n+\frac12 m^* v_n^2
$$

---

## 3. Resíduo PINN (estacionário 1D)

$$
\mathcal{L}
=
\lVert\partial_x(nv)\rVert^2
+
\left\lVert
\partial_x(nv^2)
+\frac{qn}{m^*}E
+\frac{1}{m^*}\partial_x(n k_B T_n)
+\frac{nv}{\tau_p}
\right\rVert^2
+
\left\lVert
\partial_x(vnW-\kappa\partial_x T_n)
+qn v E
+n\frac{W-W_0}{\tau_w}
\right\rVert^2
$$

---

## 4. Estrutura

```
hidrodinamico_portadores_quentes/
├── src/
│   ├── fisica_hd.py
│   ├── rede_pinn_hd.py
│   ├── residuo_hd.py
│   └── treinamento_hd.py
├── examples/hidrodinamico_hd.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/hidrodinamico_hd.py
```

---

**© 2026 Luiz Tiago Wilcke**
