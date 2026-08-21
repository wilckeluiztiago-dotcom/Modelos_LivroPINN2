# Fokker–Planck com Saltos Discretos para Memórias de Elétron Único

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo:** 8  

Equação de **Kolmogorov Forward** acoplando **difusão contínua** do potencial eletrostático \(s\) a **operadores de salto discreto** no número de elétrons \(q\in\mathbb{Z}\), para **Single-Electron Transistors (SET)** e memórias de elétron único.

---

## 1. Fenômeno Físico

1. Em SETs / pontos quânticos, a carga é **discreta** (elétron a elétron).
2. O potencial de porta/ilha \(s\) é **contínuo** e difunde termicamente.
3. As taxas de tunelamento \(\lambda^a(s),\lambda^b(s)\) geram **degraus de Coulomb**.
4. A PINN aproxima \(p(q,s,t)\) **sem malha de diferenças finitas** em \(q\).

---

## 2. Equações

### Kolmogorov Forward híbrida

$$
\frac{\partial p(q,s,t)}{\partial t}
-\frac12\sigma^2\frac{\partial^2 p}{\partial s^2}
=
\lambda^a(s)\,p(q-1,s,t)
+\lambda^b(s)\,p(q+1,s,t)
-\bigl(\lambda^a(s)+\lambda^b(s)\bigr)p(q,s,t)
$$

### Taxas de tunelamento (Coulomb)

$$
\lambda^a(s)
=
\frac{\Gamma_0}{1+\exp\bigl(\alpha(E_c(q+\tfrac12)-s-\tfrac12 V_{\mathrm{bias}})\bigr)}
$$

$$
\lambda^b(s)
=
\frac{\Gamma_0}{1+\exp\bigl(\alpha(s-E_c(q-\tfrac12)-\tfrac12 V_{\mathrm{bias}})\bigr)}
$$

### Dinâmica estocástica

- \(q \to q\pm 1\) com taxas \(\lambda^a,\lambda^b\)
- \(ds = \mu(s)\,dt + \sigma\,dW\)

---

## 3. Interpretação

| Variável | Significado |
|----------|-------------|
| \(q\in\mathbb{Z}\) | Número de elétrons aprisionados |
| \(s\) | Potencial eletrostático contínuo |
| \(\lambda^a,\lambda^b\) | Adição / remoção por tunelamento |
| \(E_c\) | Energia de carregamento (bloqueio de Coulomb) |
| \(p(q,s,t)\) | Densidade conjunta |

---

## 4. Estrutura

```
fp_saltos_set/
├── src/
│   ├── set_carga.py
│   ├── rede_pinn_set.py
│   ├── residuo_set.py
│   └── treinamento_set.py
├── examples/fp_set.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/fp_set.py
```

---

**© 2026 Luiz Tiago Wilcke**
