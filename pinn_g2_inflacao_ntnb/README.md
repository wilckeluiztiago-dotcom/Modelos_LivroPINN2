# Modelo de Inflação Implícita e Curva Real (NTN-B vs DI) via PINN de Dois Fatores (G2++)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  

Sistema de **duas taxas curtas estocásticas** (taxa real \(r_t\) e inflação instantânea \(i_t\)) no espírito **G2++**, gerando uma **EDP tridimensional** de precificação. A PINN projeta a **inflação implícita (breakeven)** ao longo da estrutura a termo, comparando **NTN-B (IPCA+)** à curva **pré (DI/LTN)**.

---

## 1. Mercado

| Instrumento | Indexador | Papel |
|-------------|-----------|--------|
| **NTN-B** / Tesouro IPCA+ | IPCA | curva **real** |
| **DI / LTN / NTN-F** | pré-fixado | curva **nominal** |
| Breakeven | implícito | \(i^{\mathrm{BE}}(\tau)\) |

---

## 2. Formulação Matemática

### Dinâmica dos fatores

$$
dr_t = \kappa_r(\theta_r - r_t)\,dt + \sigma_r\,dW^r_t
$$

$$
di_t = \kappa_i(\theta_i - i_t)\,dt + \sigma_i\,dW^i_t
$$

$$
\langle dW^r, dW^i\rangle = \rho\,dt
$$

Taxa nominal (Fisher linearizado): \(n_t \approx r_t + i_t\).

### EDP de precificação (título nominal)

$$
\frac{\partial P}{\partial\tau}
+(r+i)P
-\kappa_r(\theta_r-r)\frac{\partial P}{\partial r}
-\kappa_i(\theta_i-i)\frac{\partial P}{\partial i}
-\frac12\sigma_r^2\frac{\partial^2 P}{\partial r^2}
-\frac12\sigma_i^2\frac{\partial^2 P}{\partial i^2}
=0
$$

com \(P(r,i,0)=1\).

### Inflação implícita (breakeven)

$$
i^{\mathrm{BE}}(\tau)
=
\left(\frac{P_{\mathrm{real}}(\tau)}{P_{\mathrm{nom}}(\tau)}\right)^{1/\tau}-1
$$

---

## 3. PINN 3D

Entrada \((r,i,\tau)\) → \(P_\theta(r,i,\tau)\).  
Perda = resíduo da EDP + condição terminal.

---

## 4. Estrutura

```
pinn_g2_inflacao_ntnb/
├── src/
│   ├── modelo_g2_inflacao.py
│   ├── rede_pinn_g2.py
│   ├── residuo_g2.py
│   └── treinamento_g2.py
├── examples/g2_inflacao_ntnb.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/g2_inflacao_ntnb.py
```

---

**© 2026 Luiz Tiago Wilcke**
