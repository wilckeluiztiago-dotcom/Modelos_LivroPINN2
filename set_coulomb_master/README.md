# Bloqueio de Coulomb e Transistor de Elétron Único (SET / Equação Mestre)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

Condução através de uma **ilha quântica** onde a energia eletrostática de carregar um único elétron excede a energia térmica:

$$
E_c = \frac{e^2}{2C_\Sigma} \gg k_B T
$$

A **equação mestre** governa a densidade de probabilidade discreta \(P(N,t)\); a PINN parametriza \(P\) como distribuição normalizada.

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| **Ilha quântica** | Região isolada com capacitância \(C_\Sigma\) |
| **Bloqueio de Coulomb** | Proibição de tunelamento quando \(\Delta U > k_B T\) |
| **Degraus de Coulomb** | Oscilações de corrente vs \(V_g\) |
| **SET** | Transistor de elétron único (source–ilha–drain + gate) |

---

## 2. Equações

### Energia da ilha

$$
U(N) = E_c\bigl(N - n_g\bigr)^2,
\qquad
n_g = \frac{C_g V_g}{e}
$$

### Equação mestre

$$
\frac{\partial P(N,t)}{\partial t}
=
\sum_{N'}
\Bigl[
\Gamma(N'\to N)\,P(N',t)
-
\Gamma(N\to N')\,P(N,t)
\Bigr]
$$

Com saltos apenas \(N\leftrightarrow N\pm 1\) (tunelamento de um elétron):

$$
\frac{\partial P(N)}{\partial t}
=
\Gamma_{\mathrm{add}}(N-1)\,P(N-1)
+
\Gamma_{\mathrm{rem}}(N+1)\,P(N+1)
-
\bigl[\Gamma_{\mathrm{add}}(N)+\Gamma_{\mathrm{rem}}(N)\bigr]P(N)
$$

### Taxas (Fermi–Dirac)

$$
\Gamma_{\mathrm{add/rem}}(N)
=
\frac{\Gamma_0}{1+\exp(\Delta E_{\pm}/k_B T)}
$$

### Resíduo PINN

$$
\mathcal{R}
=
\partial_t P_\theta(N,t)
-
\sum_{N'}\bigl[\Gamma(N'\to N)P_\theta(N')-\Gamma(N\to N')P_\theta(N)\bigr]
$$

com \(P_\theta\) normalizado: \(\sum_N P_\theta(N,t)=1\).

---

## 3. Estrutura

```
set_coulomb_master/
├── src/
│   ├── fisica_set.py          # E_c, U(N), taxas
│   ├── equacao_mestre.py      # integração + varredura gate
│   ├── rede_pinn_mestre.py
│   ├── residuo_mestre.py
│   └── treinamento_mestre.py
├── examples/set_coulomb.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/set_coulomb.py
```

---

**© 2026 Luiz Tiago Wilcke**
