# Contágio Térmico por Difusão Acoplada de McKean–Vlasov em Circuitos 3D-IC

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulos:** 24 & 40  

EDP não-linear de McKean–Vlasov em que o **drift de cada elemento depende da média integral da população** \(\bar X_t=\int x\,p(x,t)\,dx\). Aplicada ao **risco de fuga térmica em cascata** em blocos densos de **GAAFETs** em 3D-IC.

---

## 1. Fenômeno Físico

Em blocos densos de nanotransistores:

1. O aquecimento **local** de um GAAFET eleva a corrente de fuga sub-limiar.
2. Essa fuga aquece vizinhos → **contágio térmico** (mean-field).
3. Acima de um limiar \(T_{\mathrm{crit}}\), a média populacional **auto-acelera** (thermal runaway).
4. A PINN resolve \(p(x,t)\) com \(\bar X_t\) obtido por **integração contínua** no grafo.

---

## 2. Equações

### EDP de McKean–Vlasov

$$
\frac{\partial p}{\partial t}
+ a\frac{\partial}{\partial x}\Bigl[\bigl(\bar X_t - x\bigr)p\Bigr]
- \frac12\sigma^2\frac{\partial^2 p}{\partial x^2}
= 0
$$

$$
\bar X_t = \int x\,p(x,t)\,dx
$$

### Sistema de partículas (interpretação)

$$
dX^i_t = a\bigl(\bar X_t - X^i_t\bigr)\,dt + \alpha\,\mathbf{1}_{\{\bar X_t>T_{\mathrm{crit}}\}}\,dt + \sigma\,dW^i_t
$$

### PINN

$$
\mathcal{J}(\theta)
=
\frac1{N_c}\sum_i\bigl|\partial_t p_\theta + a\partial_x[(\bar X_t-x)p_\theta]-\tfrac12\sigma^2\partial_{xx}p_\theta\bigr|^2
+\lambda_{\mathrm{IC}}\|p_\theta(\cdot,0)-p_0\|^2
$$

com \(\bar X_t\) recalculado por integração de \(p_\theta\) a cada avaliação.

---

## 3. Interpretação 3D-IC / GAAFET

| Símbolo | Significado |
|---------|-------------|
| \(X^i\) | Temperatura local do nanotransistor \(i\) |
| \(\bar X_t\) | Temperatura média do bloco |
| \(a(\bar X-X^i)\) | Difusão térmica acoplada (mean-field) |
| \(\alpha\mathbf{1}_{\bar X>T_{\mathrm{crit}}}\) | Auto-aceleração (fuga térmica) |
| \(p(x,t)\) | Densidade de temperaturas no bloco |

---

## 4. Estrutura

```
mckean_vlasov_termico_3dic/
├── src/
│   ├── mckean_vlasov.py    # partículas + runaway
│   ├── rede_pinn_mv.py
│   ├── residuo_mv.py       # EDP + ∫ x p dx
│   └── treinamento_mv.py
├── examples/mv_termico_3dic.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/mv_termico_3dic.py
```

---

**© 2026 Luiz Tiago Wilcke**
