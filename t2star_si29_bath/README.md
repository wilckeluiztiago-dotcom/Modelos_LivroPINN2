# Descoerência de Spin Nuclear por Acoplamento Dipolar com o Banho de ²⁹Si (T₂* Dephasing)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Em silício com concentração natural de **4.7%** de isótopos magnéticos **²⁹Si** (\(I=1/2\)), o spin nuclear do **³¹P** sofre flutuações hiperfinas pela dinâmica de **flip-flop** do banho nuclear.

---

## 1. Física

| Elemento | Papel |
|----------|--------|
| ³¹P | Qubit nuclear / doador |
| ²⁹Si (4.7%) | Banho de spins \(I=1/2\) |
| \(A_k^{\mathrm{dip}}\) | Acoplamento dipolar ³¹P–²⁹Si |
| \(T_2^*\) | Tempo de dephasing FID |

---

## 2. Equações

### Hamiltoniano do banho (dipolar + flip-flop)

$$
\mathcal{H}_{\mathrm{bath}}
=
\sum_k
\frac{\mu_0\gamma_n\gamma_{29}\hbar^2}{4\pi r_k^3}
\left[
\mathbf{I}_P\cdot\mathbf{I}_k
-
\frac{3(\mathbf{I}_P\cdot\mathbf{r}_k)(\mathbf{I}_k\cdot\mathbf{r}_k)}{r_k^2}
\right]
+
\sum_{j<k}
D_{jk}
\bigl(
I_j^+I_k^-+I_j^-I_k^+-4I_j^z I_k^z
\bigr)
$$

### FID gaussiano

$$
\langle S_x(t)\rangle
=
\exp\!\left(-(t/T_2^*)^2\right)
$$

### \(T_2^*\) a partir dos acoplamentos

$$
\frac{1}{(T_2^*)^2}
=
\frac12\sum_k\lvert A_k^{\mathrm{dipolar}}\rvert^2
$$

$$
A_k^{\mathrm{dip}}
\propto
\frac{\gamma_P\gamma_{29}}{r_k^3}(1-3\cos^2\theta_k)
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\left\lVert
\partial_t S_x + \frac{2t}{(T_2^*)^2}S_x
\right\rVert^2
+
\lambda_{T2}
\bigl\lVert
T_{2,\theta}^*
-
\bigl(\tfrac12\sum_k\lvert A_k\rvert^2\bigr)^{-1/2}
\bigr\rVert^2
+
\lambda_{\mathrm{IC}}\lvert S_x(0)-1\rvert^2
$$

---

## 4. Estrutura

```
t2star_si29_bath/
├── src/
│   ├── fisica_banho.py
│   ├── rede_pinn_t2.py
│   ├── residuo_t2.py
│   └── treinamento_t2.py
├── examples/t2star_banho.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/t2star_banho.py
```

---

**© 2026 Luiz Tiago Wilcke**
