# Precificação de Derivativos de Energia Elétrica e Opções sobre PLD via PIDE

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

**PIDE** com reversão à média **sazonal** e **saltos de hidrologia (ENA)** para precificar opções e contratos de flexibilidade no **Mercado Livre de Energia (ACL/CCEE)**, vinculados ao **PLD**.

---

## 1. Mercado

| Conceito | Descrição |
|----------|-----------|
| **PLD** | Preço de Liquidação das Diferenças (CCEE) |
| **ACL** | Ambiente de Contratação Livre |
| **ENA** | Energia Natural Afluente (hidrologia) |
| **Swing** | Opção de flexibilidade volumétrica |

---

## 2. Formulação Matemática

### Dinâmica do PLD

$$
d\ln S = k\bigl(\theta(t)-\ln S\bigr)\,dt + \sigma\,dW + \text{saltos de ENA}
$$

$$
\theta(t) = \theta_0 + A\sin(2\pi t+\phi)
\quad\text{(sazonalidade)}
$$

### PIDE

$$
\frac{\partial V}{\partial t}
+\frac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}
+k\bigl(\theta(t)-\ln S\bigr)S\frac{\partial V}{\partial S}
-rV
+\lambda\int_0^\infty\bigl[V(S\eta,t)-V(S,t)\bigr]g(\eta)\,d\eta
=0
$$

### Monte Carlo do operador integral (na PINN)

$$
\int\bigl[V(S\eta)-V(S)\bigr]g(\eta)\,d\eta
\approx
\frac1{N_{\mathrm{MC}}}\sum_{k=1}^{N_{\mathrm{MC}}}\bigl[V(S\eta_k)-V(S)\bigr]
$$

---

## 3. Estrutura

```
pide_pld_energia/
├── src/
│   ├── pld_hidrologia.py
│   ├── rede_pinn_pide.py
│   ├── residuo_pide.py
│   └── treinamento_pide.py
├── examples/pide_pld.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/pide_pld.py
```

---

**© 2026 Luiz Tiago Wilcke**
