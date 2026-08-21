# Acoplamento Espin–Langevin de Mecânica Estatística para CFETs Quânticos

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Referência:** Apêndice J.3  

Sistema acoplado **Glauber–Ising** (magnetização / ocupação discreta de sub-banda) + **Langevin** (difusão contínua do potencial de canal em potenciais não-lineares), aplicado a **Complementary-FETs** com canais empilhados verticalmente (nFET sobre pFET) separados por barreiras dielétricas **sub-nanométricas**.

---

## 1. Fenômeno Físico (Nanotransistores CFET)

Em um CFET vertical:

```
┌──────────────────┐
│      nFET        │  ocupação de sub-banda σⁿ (Ising) + potencial φₙ (Langevin)
├──────────────────┤  barreira dielétrica sub-nm  →  J_inter, κ_ep
│      pFET        │  ocupação σᵖ (Ising) + potencial φₚ (Langevin)
└──────────────────┘
```

1. O **estado quântico de ocupação** de sub-banda é modelado por uma rede de **Ising discreta** com dinâmica de **Glauber**.
2. A **distribuição contínua do potencial de canal** evolui pela **equação estocástica de Langevin** em potencial não-linear.
3. A barreira sub-nm permite **troca de spin** (\(J_{\mathrm{inter}}\)) e **acoplamento eletrostático** (\(\kappa_{\mathrm{ep}}\)) entre os canais.

---

## 2. Equações do Modelo

### Hamiltoniano de Ising (por canal)

$$
H[\sigma] = -J\sum_{\langle i,j\rangle}\sigma_i\sigma_j - h\sum_i\sigma_i,
\qquad \sigma_i\in\{-1,+1\}
$$

### Dinâmica de Glauber

$$
P(\sigma_i\to-\sigma_i) = \frac{1}{1+\exp\bigl(2\beta\sigma_i h_i\bigr)},
\qquad
h_i = J\sum_{j\sim i}\sigma_j + h_{\mathrm{eff}}
$$

Campo efetivo no CFET (acoplamento inter-canal):

$$
h_{\mathrm{eff}}^{(n)} = h_n + J_{\mathrm{inter}}\, m_p + \kappa_{\mathrm{ep}}\,\phi_p
$$

$$
h_{\mathrm{eff}}^{(p)} = h_p + J_{\mathrm{inter}}\, m_n + \kappa_{\mathrm{ep}}\,\phi_n
$$

com magnetizações

$$
m_n = \frac1N\sum_i\sigma_i^{(n)},
\qquad
m_p = \frac1N\sum_i\sigma_i^{(p)}.
$$

### Potencial não-linear de Langevin

$$
U(\phi) = \frac{a}{2}\phi^2 + \frac{b}{4}\phi^4 + c\,\phi
$$

### Equação de Langevin acoplada

$$
d\phi_n = \Bigl[-\gamma\nabla U(\phi_n) + \kappa_{\mathrm{ep}}\, m_p\Bigr]dt + \sigma\,dW_n
$$

$$
d\phi_p = \Bigl[-\gamma\nabla U(\phi_p) + \kappa_{\mathrm{ep}}\, m_n\Bigr]dt + \sigma\,dW_p
$$

### Sistema acoplado completo (Apêndice J.3)

$$
\begin{cases}
\text{Glauber:} & \sigma^{(n)},\sigma^{(p)}\ \text{com campos}\ h_{\mathrm{eff}}^{(n)}, h_{\mathrm{eff}}^{(p)} \\[4pt]
\text{Langevin:} & d\phi_n,\ d\phi_p\ \text{forçados por}\ m_p,\ m_n
\end{cases}
$$

---

## 3. Interpretação em CFET

| Variável | Significado físico |
|----------|-------------------|
| \(\sigma_i=\pm 1\) | Ocupação / vacância de estado de sub-banda |
| \(m_n, m_p\) | Polarização média de ocupação nFET / pFET |
| \(\phi_n, \phi_p\) | Potencial eletrostático contínuo de cada canal |
| \(J_{\mathrm{inter}}\) | Troca de spin através da barreira sub-nm |
| \(\kappa_{\mathrm{ep}}\) | Acoplamento eletrostático canal–canal |
| \(U(\phi)\) | Energia livre efetiva não-linear do canal |

---

## 4. Estrutura do Projeto

```
cfet_spin_langevin/
├── src/
│   ├── ising_glauber.py       # Rede Ising + dinâmica de Glauber
│   ├── langevin_potencial.py  # Langevin em U(φ) não-linear
│   ├── acoplamento_cfet.py    # CFET n/p empilhado
│   └── utils.py
├── examples/cfet_spin_langevin_sim.py
├── figures/
├── docs/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/cfet_spin_langevin_sim.py
```

---

## 6. Classes (português)

| Nome | Papel |
|------|--------|
| `RedeIsingGlauber` | Spins + passos de Glauber |
| `ProcessoLangevin` | SDE em potencial não-linear |
| `CFETSpinLangevin` | Sistema acoplado nFET/pFET |
| `potencial_nao_linear` | \(U(\phi)\) |

---

**© 2026 Luiz Tiago Wilcke**
