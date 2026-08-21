# Equação de Fokker–Planck com Potencial Bimodal de Landau para Chaveamento em TFETs

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Capítulo 41 & Apêndice J.2**

Dinâmica de Langevin em potencial quártico de Landau e equação de Fokker–Planck para modelar a **transição bistável pico–vale** em **TFETs / diodos de tunelamento ressonante** em escala **1.6 nm**. A PINN resolve a densidade de probabilidade nos dois poços e o **tempo de Kramers** quantifica a retenção frente ao ruído térmico.

---

## 1. Fenômeno Físico (TFET / RTD)

1. A característica \(I\)–\(V\) de um TFET/RTD apresenta região de **resistência diferencial negativa** (pico → vale).
2. Em escala 1.6 nm, o estado de corrente pode ser mapeado em uma **coordenada efetiva** \(x\) com **dois poços** (pico e vale).
3. O **ruído térmico** induz escapes entre poços → tempo médio de retenção \(\tau_K\).
4. A PINN fornece \(p(x,t)\) sem malha espacial rígida.

---

## 2. Equações

### Potencial de Landau

$$
V(x) = -\frac{a}{2}x^2 + \frac{b}{4}x^4
$$

Mínimos em \(x=\pm\sqrt{a/b}\), barreira em \(x=0\), altura \(\Delta V = a^2/(4b)\).

### Langevin

$$
dX_t = \bigl(a X_t - b X_t^3\bigr)\,dt + \sigma\,dW_t
$$

### Fokker–Planck

$$
\frac{\partial p}{\partial t}
=
-\frac{\partial}{\partial x}\Bigl[\bigl(ax-bx^3\bigr)p\Bigr]
+\frac{\sigma^2}{2}\frac{\partial^2 p}{\partial x^2}
$$

### Tempo de escape de Kramers

$$
\tau_K
\approx
\frac{2\pi}{\sqrt{V''(x_{\min})\,\lvert V''(x_{\max})\rvert}}
\exp\!\left(\frac{2\Delta V}{\sigma^2}\right)
$$

### Densidade estacionária

$$
p_\infty(x)\propto\exp\!\left(-\frac{2V(x)}{\sigma^2}\right)
$$

### PINN

$$
\mathcal{J}(\theta)
=
\frac1{N_c}\sum_i\bigl|\partial_t p_\theta + \partial_x(F p_\theta) - \tfrac12\sigma^2\partial_{xx}p_\theta\bigr|^2
+\lambda_{\mathrm{IC}}\frac1{N_0}\sum_j\bigl|p_\theta(x_j,0)-p_0\bigr|^2
$$

---

## 3. Interpretação em TFET 1.6 nm

| Grandeza | Significado |
|----------|-------------|
| \(x\) | Coordenada efetiva de estado (pico/vale de corrente) |
| Dois poços de \(V\) | Estados de corrente de pico e de vale |
| \(\tau_K\) | Tempo médio de retenção / imunidade ao ruído térmico |
| \(p(x,t)\) | Probabilidade de ocupação dos estados quânticos efetivos |

---

## 4. Estrutura

```
fp_landau_tfet/
├── src/
│   ├── potencial_landau.py   # V, Kramers
│   ├── langevin_fp.py        # SDE + p_∞
│   ├── rede_pinn_fp.py
│   ├── residuo_fp.py
│   └── treinamento_fp.py
├── examples/fp_landau_tfet.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/fp_landau_tfet.py
```

---

**© 2026 Luiz Tiago Wilcke**
