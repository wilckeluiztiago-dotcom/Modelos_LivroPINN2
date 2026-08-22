# Diodo de Tunelamento Ressonante de Dupla Barreira (DBRTD / Tsu–Esaki)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Condução quântica através de poços ultra-finos **AlGaAs/GaAs/AlGaAs**: a transmissão atinge picos unitários quando a energia incidente ressoa com estados quase-ligados do poço, gerando **resistência diferencial negativa (NDR)**.

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| Dupla barreira | Duas barreiras + poço central |
| Ressonância | \(\mathcal{T}(E_{\mathrm{res}})\to 1\) |
| NDR | \(dJ/dV < 0\) em trechos da curva J–V |
| Tsu–Esaki | Corrente integrada com fator logarítmico |

---

## 2. Equações

### Schrödinger 1D

$$
\left[
-\frac{\hbar^2}{2}
\frac{d}{dx}
\left(\frac{1}{m^*(x)}\frac{d}{dx}\right)
+
V(x)-E
\right]
\psi(x)=0
$$

### Corrente Tsu–Esaki

$$
J(V)
=
\frac{q m^* k_B T}{2\pi^2\hbar^3}
\int_0^\infty
\mathcal{T}(E_x,V)
\ln
\left(
\frac{1+\exp[(E_F-E_x)/k_B T]}
{1+\exp[(E_F-E_x-qV)/k_B T]}
\right)
dE_x
$$

---

## 3. Resíduo PINN

Rede \(\psi(x,E)=\psi_R+i\psi_I\) com BC de onda aberta:

$$
\mathcal{L}
=
\bigl\lVert\hat H_{\mathrm{DB}}\psi-E\psi\bigr\rVert^2
+
\left\lVert
\bigl(\partial_x\psi-ik_L\psi\bigr)_{x=0}-2ik_L
\right\rVert^2
+
\left\lVert
\bigl(\partial_x\psi+ik_R\psi\bigr)_{x=L}
\right\rVert^2
$$

---

## 4. Estrutura

```
dbrtd_tsu_esaki/
├── src/
│   ├── fisica_dbrtd.py
│   ├── rede_pinn_dbrtd.py
│   ├── residuo_dbrtd.py
│   └── treinamento_dbrtd.py
├── examples/dbrtd_tsu_esaki.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/dbrtd_tsu_esaki.py
```

---

**© 2026 Luiz Tiago Wilcke**
