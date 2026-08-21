# PINN Integro-Diferencial (PIDE) para Tunelamento Source–Drain

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Ano:** 2026  

Modelagem da **fuga de corrente por penetração de barreira quântica** em canais **sub-12 nm**, via **PINN integro-diferencial**: amostragem de **Monte Carlo contínua** acoplada ao **operador diferencial de continuidade**.

---

## 1. Fenômeno Físico

Em transistores com \(L_{\mathrm{ch}} < 12\,\mathrm{nm}\):

1. A barreira eletrostática source–drain torna-se suficientemente estreita para **tunelamento quântico** apreciável.
2. A corrente de fuga \(I_{\mathrm{off}}\) deixa de ser apenas termiônica (sobre a barreira) e passa a incluir **penetração através da barreira**.
3. A descrição local de deriva–difusão é incompleta: é necessário um **termo não-local** (integral) que acopla ocupações em \(x\) e \(y\) através do kernel de tunelamento.
4. O resultado é uma **equação integro-diferencial (PIDE)** para a densidade \(n(x,t)\).

---

## 2. Equações do Modelo

### Continuidade com fonte de tunelamento

$$
\frac{\partial n}{\partial t} + \frac{\partial J}{\partial x} = G_{\mathrm{tun}}[n]
$$

### Corrente de deriva–difusão

$$
J = \mu\, n\, E - D\,\frac{\partial n}{\partial x}
$$

### Operador integral de tunelamento

$$
G_{\mathrm{tun}}[n](x)
=
\int_0^{L} K(x,y)\,\bigl(n(y)-n(x)\bigr)\,dy
$$

### Kernel de penetração de barreira (WKB)

$$
K(x,y)
=
\exp\!\Biggl(
-\alpha\int_{\min(x,y)}^{\max(x,y)}
\kappa(s)\,ds
\Biggr),
\qquad
\kappa(s)=\frac{\sqrt{2m^*\bigl(V(s)-E\bigr)_+}}{\hbar}
$$

### Transmissão WKB da barreira

$$
T(E)\approx\exp\!\Biggl(-2\int_{x_1}^{x_2}\kappa(x)\,dx\Biggr)
$$

### Forma estacionária (resíduo PIDE)

$$
\frac{dJ}{dx} - G_{\mathrm{tun}}[n] = 0
$$

---

## 3. Amostragem de Monte Carlo contínua

O integral em \(G_{\mathrm{tun}}\) é estimado **sem malha** por Monte Carlo:

$$
G_{\mathrm{tun}}[n](x)
\approx
\frac{L}{N_{\mathrm{MC}}}
\sum_{j=1}^{N_{\mathrm{MC}}}
K(x,y_j)\,\bigl(n(y_j)-n(x)\bigr),
\qquad
y_j\sim\mathrm{Unif}[0,L]
$$

Assim, o operador integral é **acoplado** ao operador diferencial \(\partial_x J\) dentro da mesma perda PINN.

---

## 4. Perda composta da PINN–PIDE

$$
\mathcal{J}(\theta)
=
\frac{1}{N_c}\sum_{i=1}^{N_c}
\Biggl|
\frac{dJ_\theta}{dx}(x_i) - G_{\mathrm{tun}}[n_\theta](x_i)
\Biggr|^2
+
\lambda_{\mathrm{BC}}
\frac{1}{N_b}\sum_{b}
\bigl|n_\theta(x_b)-n_b\bigr|^2
$$

onde \(n_\theta=\mathrm{PINN}_\theta(x)\) e \(J_\theta=\mu n_\theta E - D\,\partial_x n_\theta\).

---

## 5. Estrutura do Projeto

```
pinn_pide_tunelamento_sd/
├── src/
│   ├── barreira_tunelamento.py   # V(x), κ, T_WKB, kernel K
│   ├── rede_pinn.py
│   ├── residuo_pide.py           # continuidade + MC integral
│   ├── treinamento.py
│   └── utils.py
├── examples/pide_tunelamento_sd.py
├── figures/
├── docs/
├── artigo/
└── README.md
```

---

## 6. Uso

```bash
pip install -r requirements.txt
python examples/pide_tunelamento_sd.py
```

---

## 7. Classes (português)

| Nome | Papel |
|------|--------|
| `CanalSub12nm` | Barreira e WKB sub-12 nm |
| `kernel_tunelamento` | Kernel \(K(x,y)\) |
| `operador_tunelamento_mc` | \(G_{\mathrm{tun}}\) via Monte Carlo |
| `residuo_pide_estacionario` | \(\partial_x J - G_{\mathrm{tun}}\) |
| `treinar_pide` | Otimização da perda composta |

---

**© 2026 Luiz Tiago Wilcke**
