# Market Making de Alta Frequência em Mini-Índice (WIN) e Mini-Dólar (WDO) via PI-DRL

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

Controle estocástico **Avellaneda–Stoikov** para market making no livro de ofertas (LOB) dos futuros **WIN** e **WDO** na B3, com agente **Physics-Informed Deep Reinforcement Learning (PI-DRL)** cujo Critic é regularizado pelo resíduo da equação **HJB**.

---

## 1. Mercado

| Contrato | Underlying | LOB |
|----------|------------|-----|
| **WIN** | Mini-Ibovespa | B3 |
| **WDO** | Mini-Dólar | B3 |

---

## 2. Formulação Matemática (Avellaneda–Stoikov)

### HJB de market making

$$
\frac{\partial v}{\partial t}
+\frac12\gamma^2\sigma^2 q^2 v
+\max_{\delta^b}\Bigl[\lambda^b(\delta^b)\bigl(v(q+1)e^{\gamma\delta^b}-v\bigr)\Bigr]
+\max_{\delta^a}\Bigl[\lambda^a(\delta^a)\bigl(v(q-1)e^{\gamma\delta^a}-v\bigr)\Bigr]
=0
$$

### Intensidade de chegada

$$
\lambda(\delta)=A\,e^{-k\delta}
$$

### Reservas e spread ótimo

$$
r=s-q\gamma\sigma^2(T-t)-\frac1\gamma\ln\Bigl(1+\frac\gamma k\Bigr)
$$

$$
\delta^*=\frac1\gamma\ln\Bigl(1+\frac\gamma k\Bigr)+\frac12\gamma\sigma^2(T-t)
$$

---

## 3. PI-DRL

| Componente | Papel |
|------------|--------|
| **Actor** | política \(\delta^b,\delta^a\) a partir de \((t,s,q)\) |
| **Critic** | \(v_\theta(t,s,q)\) |
| **Perda Critic** | TD-error + \(\lambda_{\mathrm{HJB}}\|\mathcal{R}_{\mathrm{HJB}}\|^2\) |

A regularização HJB evita **overshooting de inventário** em regimes de alta volatilidade.

---

## 4. Estrutura

```
pi_drl_mm_win_wdo/
├── src/
│   ├── avellaneda_stoikov.py
│   ├── rede_pidrl.py
│   ├── residuo_hjb_mm.py
│   └── treinamento_pidrl.py
├── examples/pidrl_mm.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/pidrl_mm.py
```

---

**© 2026 Luiz Tiago Wilcke**
