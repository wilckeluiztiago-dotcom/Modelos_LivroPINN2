# Contágio Fermi–Dirac em Cadeias de Dopantes Discretos para Single-Electron FETs

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Apêndice:** J.4  

Dinâmica de redes estocásticas com probabilidades de transição **Fermi–Dirac**, aplicada a **bloqueio de Coulomb** e transporte **elétron-a-elétron** em nanofios ultra-estreitos, com **PINN** para o mapa de oscilações de condutância — **sem discretização espacial** contínua.

---

## 1. Fenômeno Físico (SEFET)

Em transistores de elétron único (Single-Electron FETs) com dopantes discretos:

1. Cada dopante atômico é um **sítio de localização** com energia \(D_i\).
2. O **bloqueio de Coulomb** impede ocupação múltipla no mesmo sítio (regime \(n_i\in\{0,1\}\)).
3. O transporte ocorre por **saltos correlacionados** elétron-a-elétron entre sítios.
4. A varredura de gate produz **oscilações de condutância** (picos de Coulomb).

---

## 2. Equações do Modelo

### Probabilidade de transição Fermi–Dirac

$$
P_{ij}
=
\frac{1}{1+\exp\bigl(\beta(D_i-D_j)\bigr)}
$$

- \(D_i\): energia efetiva do sítio destino  
- \(D_j\): energia do sítio origem  
- \(\beta=1/(k_BT)\): inverso da temperatura  

Quando \(D_i<D_j\) (destino mais baixo), \(P_{ij}\to 1\).

### Energia efetiva com Coulomb

$$
D_i^{\mathrm{eff}}
=
D_i^{(0)} + U\sum_{j\neq i}\frac{n_j}{|x_i-x_j|+\lambda}
$$

### Dinâmica de contágio (rede estocástica)

1. **Injeção** source \(\to\) sítio 0 com \(P\) Fermi–Dirac  
2. **Saltos** entre vizinhos \(i\leftrightarrow i+1\) com \(P_{ij}\)  
3. **Ejeção** sítio \(N\) \(\to\) drain  

### Corrente e condutância

$$
I(t)=\sum_{\text{ejeções em }[t,t+\Delta t]} e,
\qquad
G(V_g)=\langle I\rangle_{V_g}
$$

### PINN para o mapa \(G(V_g)\)

$$
\mathcal{J}(\theta)
=
\frac1N\sum_k\bigl|G_\theta(V_g^{(k)})-G_k^{\mathrm{sim}}\bigr|^2
+\lambda_{\mathrm{suave}}\|\partial_{V}G_\theta\|^2
$$

Aproxima o regime de oscilação de Coulomb **sem malha espacial**.

---

## 3. Estrutura

```
fermi_dirac_contagio_sefet/
├── src/
│   ├── cadeia_dopantes.py       # Sítios, Coulomb, ocupação
│   ├── contagio_fermi_dirac.py  # P_ij e dinâmica de saltos
│   ├── pinn_condutancia.py      # PINN G(V_g)
│   └── utils.py
├── examples/sefet_fermi_dirac.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/sefet_fermi_dirac.py
```

---

## 5. Classes (português)

| Nome | Papel |
|------|--------|
| `CadeiaDopantes` | Sítios discretos + Coulomb |
| `probabilidade_fermi_dirac` | \(P_{ij}\) |
| `simular_transporte` | Dinâmica elétron-a-elétron |
| `RedePINN` / `treinar_condutancia` | Mapa \(G(V_g)\) |

---

**© 2026 Luiz Tiago Wilcke**
