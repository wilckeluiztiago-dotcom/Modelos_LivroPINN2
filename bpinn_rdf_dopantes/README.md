# Bayesian PINNs (B-PINNs) para Incerteza Epistêmica em Flutuações de Dopantes (RDF)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Apêndice:** C.3  

Formulação **Bayesiana de PINNs** com prior sobre os pesos e quantificação de **incerteza epistêmica**, aplicada à predição do **potencial eletrostático** em canais de **1.6 nm** sob **Random Dopant Fluctuations (RDF)**.

---

## 1. Fenômeno Físico

Em canais de \(L_z\sim 1{,}6\,\mathrm{nm}\):

1. Poucos dopantes atômicos → a **posição aleatória** de cada átomo altera fortemente \(\rho(x)\) e \(\phi(x)\).
2. Uma única solução determinística de Poisson **não captura** a variabilidade de processo.
3. B-PINNs fornecem **média**, **variância** e **intervalos de confiança** para \(\phi\).

---

## 2. Equações

### Poisson

$$
-\varepsilon\,\partial_{xx}\phi = \rho_{\mathrm{RDF}}(x),
\qquad
\rho_{\mathrm{RDF}}(x)=\sum_{k=1}^{N_d} q\,\exp\!\Bigl(-\frac{(x-x_k)^2}{2\ell^2}\Bigr)
$$

com \(x_k\) posições aleatórias dos dopantes.

### Formulação Bayesiana da PINN

Prior sobre pesos:

$$
p(\theta)=\mathcal{N}(0,\sigma_0^2 I)
$$

Verossimilhança via resíduo e contorno:

$$
p(\mathcal{D}\mid\theta)\propto\exp\Bigl(-\frac{\|\mathcal{R}_\theta\|^2}{2\sigma^2}-\frac{\|\phi_\theta-g\|_{\partial}^2}{2\sigma_{\mathrm{bc}}^2}\Bigr)
$$

Posterior preditivo:

$$
p(\phi(x)\mid\mathcal{D})
=\int p(\phi(x)\mid\theta)\,p(\theta\mid\mathcal{D})\,d\theta
$$

### Aproximação por ensemble (prática)

Treina-se \(M\) PINNs com sementes distintas; média e variância empíricas:

$$
\hat\mu(x)=\frac1M\sum_{m=1}^M\phi_{\theta_m}(x),
\qquad
\hat\sigma^2(x)=\frac1M\sum_{m=1}^M\bigl(\phi_{\theta_m}(x)-\hat\mu(x)\bigr)^2
$$

Intervalo de confiança aproximado 95%:

$$
\hat\mu(x)\pm 1{,}96\,\hat\sigma(x)
$$

### ELBO (formulação variacional completa)

$$
\mathcal{L}_{\mathrm{ELBO}}
=
\mathbb{E}_{q(\theta)}\bigl[\log p(\mathcal{D}\mid\theta)\bigr]
-\mathrm{KL}\bigl(q(\theta)\|p(\theta)\bigr)
$$

(o código inclui também a rede variacional diagonal; o exemplo principal usa ensemble por estabilidade numérica.)

---

## 3. Estrutura

```
bpinn_rdf_dopantes/
├── src/
│   ├── rdf_dopantes.py        # Canal + dopantes aleatórios
│   ├── ensemble_bpinn.py      # Ensemble B-PINN
│   ├── rede_bpinn.py          # Posterior variacional
│   └── residuo_bpinn.py
├── examples/bpinn_rdf.py
├── figures/
├── artigo/
└── README.md
```

---

## 4. Uso

```bash
pip install -r requirements.txt
python examples/bpinn_rdf.py
```

---

**© 2026 Luiz Tiago Wilcke**
