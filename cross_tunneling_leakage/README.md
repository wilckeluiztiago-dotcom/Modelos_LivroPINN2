# Corrente de Fuga por Tunelamento Quântico Inter-Fio (Direct Cross-Tunneling Leakage)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Quando dois **nanofios vizinhos** são polarizados em potenciais diferentes sob dielétricos ultra-finos (**sub-1 nm**), elétrons tunelam transversalmente por emissão quântica direta, drenando carga do circuito lógico.

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| \(d_{\mathrm{int}}\) | Distância inter-fio (barreira dielétrica) |
| \(\Phi_B\) | Altura da barreira de potencial |
| \(J_{\mathrm{leak}}\) | Densidade de corrente de tunelamento |
| Linhas 1 e 2 | Modeladas como linhas de transmissão RLGC |

---

## 2. Equações

### Corrente de tunelamento (WKB / forma fechada)

$$
J_{\mathrm{leak}}(z,t)
=
\frac{q^2\sqrt{2m^*\Phi_B}}{h^2 d_{\mathrm{int}}}
\bigl[V_1(z,t)-V_2(z,t)\bigr]
\exp\!\left(
-\frac{4\pi d_{\mathrm{int}}}{h}\sqrt{2m^*\Phi_B}
\right)
$$

Forma efetiva:

$$
J_{\mathrm{leak}} = G_{\mathrm{eff}}(d_{\mathrm{int}},\Phi_B)\,(V_1-V_2)
$$

### Continuidade de carga nos dois fios

$$
\frac{\partial I_1}{\partial z} + C_1\frac{\partial V_1}{\partial t} = -J_{\mathrm{leak}}
$$

$$
\frac{\partial I_2}{\partial z} + C_2\frac{\partial V_2}{\partial t} = +J_{\mathrm{leak}}
$$

### Equações de telegrafista (indutância + resistência)

$$
\frac{\partial V_k}{\partial z} + L_k\frac{\partial I_k}{\partial t} + R_k I_k = 0,
\qquad k=1,2
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\sum_{k=1}^{2}
\left\lVert
\partial_z I_k + C_k\partial_t V_k - (-1)^k J_{\mathrm{leak}}
\right\rVert^2
+
\sum_{k=1}^{2}
\left\lVert
\partial_z V_k + L_k\partial_t I_k + R_k I_k
\right\rVert^2
+
\lambda_{\mathrm{BC}}
\lVert V-V_{\mathrm{BC}}\rVert^2
$$

Rede \(f_\theta(z,t)=(V_1,V_2,I_1,I_2)\) com **autograd** PyTorch.

---

## 4. Estrutura

```
cross_tunneling_leakage/
├── src/
│   ├── fisica_tunel.py
│   ├── rede_pinn_tunel.py
│   ├── residuo_tunel.py
│   └── treinamento_tunel.py
├── examples/cross_tunnel.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/cross_tunnel.py
```

---

**© 2026 Luiz Tiago Wilcke**
