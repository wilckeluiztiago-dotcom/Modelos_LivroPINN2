# Feynman–Kac Estocástico com Saltos para Tunelamento Assistido por Armadilhas (TAT)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo 17 & Apêndice A.7**

Representação **integro-diferencial de Feynman–Kac** para processos contínuos pontuados por **saltos de Poisson**, aplicada ao **tunelamento assistido por armadilhas (TAT)** em dielétricos de porta **HfO₂/ZrO₂** ultra-finos (~1.6 nm).

---

## 1. Fenômeno Físico

1. Em HfO₂/ZrO₂, defeitos atômicos formam **armadilhas** de elétrons.
2. **Poole–Frenkel**: emissão térmica clássica na banda (difusão).
3. **Tunelamento entre armadilhas**: saltos quânticos espaciais (operador integral).
4. A PINN resolve a PIDE com o integral estimado por **Monte Carlo contínuo**.

---

## 2. Equações

### Processo de jump-diffusion

$$
\frac{dS}{S}
=
(r-\lambda\kappa)\,dt
+\sigma\,dW
+(\eta-1)\,dN_t
$$

### PIDE de Feynman–Kac

$$
\frac{\partial V}{\partial t}
+\frac12\sigma^2 S^2\frac{\partial^2 V}{\partial S^2}
+(r-\lambda\kappa)S\frac{\partial V}{\partial S}
-(r+\lambda)V
+\lambda\int_0^\infty V(S\eta,t)\,g(\eta)\,d\eta
=0
$$

### Densidade de salto

$$
g(\eta)
=
\frac{1}{\eta\,\sigma_j\sqrt{2\pi}}
\exp\!\Bigl(-\frac{(\ln\eta-\mu_j)^2}{2\sigma_j^2}\Bigr)
$$

### Monte Carlo do operador integral

$$
\int V(S\eta,t)\,g(\eta)\,d\eta
\approx
\frac1{N_{\mathrm{MC}}}\sum_{k=1}^{N_{\mathrm{MC}}} V(S\eta_k,t),
\qquad\eta_k\sim g
$$

---

## 3. Interpretação TAT

| Símbolo | Significado físico |
|---------|-------------------|
| \(S\) | Energia / ocupação efetiva do portador na armadilha |
| Difusão \(\sigma\) | Agitação térmica Poole–Frenkel |
| Saltos \(\lambda,g\) | Tunelamento quântico entre defeitos vizinhos |
| \(V(S,t)\) | Funcional de Feynman–Kac (probabilidade / valor esperado) |

---

## 4. Estrutura

```
feynman_kac_tat/
├── src/
│   ├── processo_saltos.py
│   ├── tat_dieletrico.py
│   ├── rede_pinn_fk.py
│   ├── residuo_fk.py          # PIDE + MC integral
│   └── treinamento_fk.py
├── examples/fk_tat.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/fk_tat.py
```

---

**© 2026 Luiz Tiago Wilcke**
