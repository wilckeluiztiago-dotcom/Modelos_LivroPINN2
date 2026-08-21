# Modelo de Volatilidade Local-Estocástica (LSV) e Condição de Gyöngy para Mobilidade de Portadores

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo:** 21  

Adaptação da dinâmica híbrida **Dupire–Heston** à **mobilidade efetiva de elétrons** em canais **sub-2 nm**, com calibração pela **condição de Gyöngy** via PINN.

---

## 1. Fenômeno Físico (Nanotransistores sub-2 nm)

Sob campos elétricos extremos:

1. A **velocidade de saturação** depende do campo longitudinal \(E_x\) de forma não-linear.
2. **Flutuações rápidas de espalhamento fonônico** introduzem um fator estocástico de variância \(\nu_t\) (processo CIR).
3. O produto **local × estocástico** define a mobilidade/velocidade efetiva observada.
4. A **condição de Gyöngy** garante que a volatilidade local \(L(E,t)\) reproduza a mobilidade efetiva média \(\mu_{\mathrm{eff}}(E,t)\), conservando a corrente média.

---

## 2. Dinâmica LSV (Dupire–Heston adaptada)

### Forma financeira original

$$
dS_t = r S_t\,dt + L(S_t,t)\sqrt{\nu_t}\,S_t\,dW_t^1
$$

$$
d\nu_t = \kappa(\theta-\nu_t)\,dt + \xi\sqrt{\nu_t}\,dW_t^2
$$

### Adaptação a portadores (campo / velocidade efetiva)

$$
dE_t = \mu_E\,dt + L(E_t,t)\sqrt{\nu_t}\,dW_t^1
$$

$$
d\nu_t = \kappa(\theta-\nu_t)\,dt + \xi\sqrt{\nu_t}\,dW_t^2
\quad\text{(CIR / fonons)}
$$

### Fator local e saturação de velocidade

$$
\mu_{\mathrm{eff}}(E)
=
\frac{\mu_0}{\bigl(1+(E/E_{\mathrm{sat}})^\beta\bigr)^{1/\beta}}
\qquad\text{(Caughey–Thomas)}
$$

$$
v(E) = v_{\mathrm{sat}}\frac{E}{E+E_{\mathrm{sat}}}
$$

---

## 3. Condição de Gyöngy (calibração)

$$
L^2(E,t)\cdot\mathbb{E}[\nu_t\mid E_t=E] = \mu_{\mathrm{eff}}^2(E,t)
$$

Logo

$$
L(E,t) = \frac{\mu_{\mathrm{eff}}(E,t)}{\sqrt{\mathbb{E}[\nu_t\mid E_t=E]}}
$$

A média condicional \(\mathbb{E}[\nu\mid E]\) é estimada por kernel a partir de trajetórias LSV; \(L(E)\) é refinada por uma **PINN** que minimiza

$$
\mathcal{J}(\theta)
=
\frac1N\sum_i\Bigl(L_\theta(E_i)^2\cdot\mathbb{E}[\nu\mid E_i] - \mu_{\mathrm{eff}}^2(E_i)\Bigr)^2
$$

garantindo a **conservação da corrente média** (analogia à match da variância local no sentido de Gyöngy / Fokker–Planck).

---

## 4. Interpretação em canal sub-2 nm

| Símbolo | Significado físico |
|---------|-------------------|
| \(E_t\) | Campo longitudinal efetivo / proxy de velocidade |
| \(\nu_t\) | Intensidade de espalhamento fonônico (CIR) |
| \(L(E,t)\) | Fator local de mobilidade (Dupire) |
| \(\mu_{\mathrm{eff}}(E)\) | Mobilidade com saturação de velocidade |
| Gyöngy | Calibração \(L\) ↔ \(\mu_{\mathrm{eff}}\) com conservação média |

---

## 5. Estrutura

```
lsv_gyongy_mobilidade/
├── src/
│   ├── cir_variancia.py        # Processo CIR (fonons)
│   ├── fator_local.py          # L(E), μ_eff, v(E)
│   ├── lsv_dinamica.py         # Dinâmica híbrida LSV
│   ├── gyongy_calibracao.py    # Condição de Gyöngy
│   ├── rede_pinn_gyongy.py     # PINN para L(E)
│   └── treinamento_gyongy.py
├── examples/lsv_gyongy_mobilidade.py
├── figures/
├── artigo/
└── README.md
```

---

## 6. Uso

```bash
pip install -r requirements.txt
python examples/lsv_gyongy_mobilidade.py
```

---

**© 2026 Luiz Tiago Wilcke**
