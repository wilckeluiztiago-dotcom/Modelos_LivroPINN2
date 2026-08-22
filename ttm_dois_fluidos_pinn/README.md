# Modelo de Dois Fluidos Elétron–Fônon (Two-Temperature Model — TTM)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Seção:** II. Efeitos Térmicos, Transporte Não-Fourier e Autoaquecimento — Item 9

Não-equilíbrio térmico extremo em canais de **~1 nm**, onde os **elétrons superaquecem** antes de transferir energia à **rede cristalina** (fônons) via acoplamento \(G_{e\text{-}ph}\).

---

## 1. Fenômeno Físico

| Grandeza | Papel |
|----------|--------|
| \(T_e(x,t)\) | Temperatura dos elétrons (hot carriers) |
| \(T_L(x,t)\) | Temperatura da rede (fônons) |
| \(G_{e\text{-}ph}\) | Taxa de troca de energia elétron–fônon |
| \(J\cdot\mathcal{E}\) | Aquecimento Joule (autoaquecimento) |
| \(\kappa_e,\kappa_L\) | Condutividades térmicas |

Em nanotransistores, \(T_e \gg T_L\) em regimes de campo alto — o TTM captura esse **desacoplamento térmico**.

---

## 2. Equações do Modelo

### Balanço de energia dos elétrons

$$
C_e \frac{\partial T_e}{\partial t}
=
\nabla \cdot \bigl( \kappa_e \nabla T_e \bigr)
-
G_{e\text{-}ph}\,(T_e - T_L)
+
J \cdot \mathcal{E}
$$

### Balanço de energia da rede (fônons)

$$
C_L \frac{\partial T_L}{\partial t}
=
\nabla \cdot \bigl( \kappa_L \nabla T_L \bigr)
+
G_{e\text{-}ph}\,(T_e - T_L)
$$

### Fonte Joule (autoaquecimento)

$$
J \cdot \mathcal{E}
\;\approx\;
\sigma \,\lvert\nabla\phi\rvert^2
\;=\;
\sigma \,\lvert\mathcal{E}\rvert^2
$$

### Forma unidimensional (implementação)

$$
C_e\,\partial_t T_e
=
\kappa_e\,\partial_{xx} T_e
-
G\,(T_e - T_L)
+
\sigma E^2
$$

$$
C_L\,\partial_t T_L
=
\kappa_L\,\partial_{xx} T_L
+
G\,(T_e - T_L)
$$

---

## 3. Resíduo PINN

A rede \(f_\theta(x,t)=(T_e^\theta,\,T_L^\theta)\) minimiza:

$$
\mathcal{L}
=
\Bigl\lVert
C_e\,\partial_t T_e
-
\nabla\cdot(\kappa_e\nabla T_e)
+
G(T_e-T_L)
-
\sigma\lvert\nabla\phi\rvert^2
\Bigr\rVert^2
+
\Bigl\lVert
C_L\,\partial_t T_L
-
\nabla\cdot(\kappa_L\nabla T_L)
-
G(T_e-T_L)
\Bigr\rVert^2
+
\lambda_{\mathrm{IC}}
\Bigl(
\lVert T_e(x,0)-T_e^0\rVert^2
+
\lVert T_L(x,0)-T_L^0\rVert^2
\Bigr)
$$

---

## 4. Estrutura do Projeto

```
ttm_dois_fluidos_pinn/
├── src/
│   ├── fisica_ttm.py          # parâmetros, FD 1D, simulação
│   ├── rede_pinn_ttm.py       # (x,t) → (T_e, T_L)
│   ├── residuo_ttm.py         # R_e, R_L
│   └── treinamento_ttm.py
├── examples/ttm_dois_fluidos.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/ttm_dois_fluidos.py
```

---

## 6. Interpretação em Nanotransistores

1. Campo elétrico alto → **Joule** deposita energia nos elétrons.
2. \(C_e \ll C_L\) → \(T_e\) sobe rápido; \(T_L\) responde atrasada.
3. Acoplamento \(G_{e\text{-}ph}\) equaliza as temperaturas em escala \(C_e/G\).
4. A PINN resolve o sistema acoplado **sem malha** rígida, útil para canais sub-2 nm.

---

**© 2026 Luiz Tiago Wilcke**
