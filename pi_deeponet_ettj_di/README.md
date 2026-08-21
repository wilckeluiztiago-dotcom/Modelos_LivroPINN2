# Dinâmica da Curva de Juros (ETTJ DI Futuro B3) via PI-DeepONet

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

**PI-DeepONet** para a **estrutura a termo de taxas de juros (ETTJ)** dos contratos futuros de **Depósito Interfinanceiro (DI1)** na **B3**, com convenção de **252 dias úteis** e **capitalização composta**.

---

## 1. Mercado e Convenção B3

- Instrumento: **DI1** (Depósito Interfinanceiro futuro)
- Dias úteis: \(\tau = \mathrm{DU}/252\)
- Capitalização composta:
  $$
  \mathrm{DF}(\tau) = \frac{1}{(1+r)^{\tau}}
  $$
- Vértices líquidos típicos: DI1F26, DI1F27, DI1F29, …

---

## 2. Formulação Matemática (HJM / PDE do título)

Operador de não-arbitragem que mapeia a curva inicial de taxas forward \(f(0,\cdot)\) na superfície de preços \(P(t,T)\):

$$
\frac{\partial P}{\partial t} + r_t P - f(t,T)P + \frac12\sigma_P^2(t,T)P = 0
$$

com condição terminal \(P(T,T)=1\).

Em curva estática (\(\sigma\to 0\)):

$$
\frac{\partial P}{\partial t} + r(t)\,P \approx 0
$$

---

## 3. PI-DeepONet

$$
P_\theta\bigl(f(0,\cdot);\,t,T\bigr)
=
\sum_{k=1}^{p} b_k\bigl(f(0,\cdot)\bigr)\,t_k(t,T) + \beta
$$

| Rede | Entrada | Papel |
|------|---------|--------|
| **Branch** | vértices DI \((r_{\tau_1},\ldots,r_{\tau_n})\) | embute a curva |
| **Trunk** | \((t,T)\) | interpola tempo/maturidade |

A marcação a mercado de **toda** a ETTJ é instantânea: muda-se a Branch (novos vértices) sem retreinar a Trunk.

### Perda composta

$$
\mathcal{J}
=
\|P_\theta - P^{\mathrm{mkt}}\|^2
+\lambda_{\mathrm{PDE}}\|\partial_t P_\theta + r_t P_\theta\|^2
+\lambda_{T}\|P_\theta(T,T)-1\|^2
$$

---

## 4. Estrutura

```
pi_deeponet_ettj_di/
├── src/
│   ├── curva_di_b3.py          # convenção 252, DF, superfície P
│   ├── rede_deeponet.py        # Branch + Trunk
│   ├── residuo_hjm.py          # PDE de não-arbitragem
│   └── treinamento_deeponet.py
├── examples/deeponet_ettj_di.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/deeponet_ettj_di.py
```

---

**© 2026 Luiz Tiago Wilcke**
