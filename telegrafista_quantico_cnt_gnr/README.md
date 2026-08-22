# Linha de Transmissão Quântica para Nanotubos de Carbono e Fitas de Grafeno (Telegrafista Quântico)

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Fios moleculares como **CNTs** e **GNRs** possuem **indutância cinética** e **capacitância quântica** que superam os efeitos magnéticos e eletrostáticos clássicos:

$$
L_K \approx \frac{h}{4e^2 v_F} \sim 8\,\mathrm{nH/\mu m},
\qquad
C_Q \approx \frac{4e^2}{h v_F} \sim 100\,\mathrm{aF/\mu m}.
$$

---

## 1. Física

| Parâmetro | Origem | Escala típica |
|-----------|--------|----------------|
| \(L_K\) | Inércia dos portadores (cinética) | ~8 nH/μm |
| \(L_{\mathrm{mag}}\) | Indutância magnética clássica | ≪ \(L_K\) |
| \(C_Q\) | Densidade de estados 1D | ~100 aF/μm |
| \(C_{\mathrm{es}}\) | Capacitância eletrostática | frequentemente em série com \(C_Q\) |

---

## 2. Equações do Telegrafista Quântico

### Queda de tensão (indutância total)

$$
\frac{\partial V(z,t)}{\partial z}
=
-R_{\mathrm{dist}} I(z,t)
-
\bigl(L_{\mathrm{mag}}+L_K\bigr)\frac{\partial I(z,t)}{\partial t}
$$

### Continuidade de corrente (capacitância efetiva)

$$
\frac{\partial I(z,t)}{\partial z}
=
-G_{\mathrm{dist}} V(z,t)
-
C_{\mathrm{eff}}\frac{\partial V(z,t)}{\partial t}
$$

com

$$
C_{\mathrm{eff}}
=
\left(
\frac{1}{C_{\mathrm{es}}}
+
\frac{1}{C_Q}
\right)^{-1}.
$$

### Impedância e velocidade

$$
Z_0 \approx \sqrt{\frac{L_{\mathrm{tot}}}{C_{\mathrm{eff}}}},
\qquad
v \approx \frac{1}{\sqrt{L_{\mathrm{tot}} C_{\mathrm{eff}}}}
\quad(\text{pode aproximar } v_F).
$$

### Condição de carga

$$
V(L,t)=Z_L\,I(L,t).
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\bigl\lVert
\partial_z V + R I + L_{\mathrm{tot}}\partial_t I
\bigr\rVert^2
+
\bigl\lVert
\partial_z I + G V + C_{\mathrm{eff}}\partial_t V
\bigr\rVert^2
+
\lambda_{\mathrm{src}}\lVert V(0,t)-V_{\mathrm{src}}(t)\rVert^2
+
\lambda_{\mathrm{term}}\lVert V(L,t)-Z_L I(L,t)\rVert^2
$$

---

## 4. Estrutura

```
telegrafista_quantico_cnt_gnr/
├── src/
│   ├── fisica_telegrafista.py
│   ├── rede_pinn_qtl.py
│   ├── residuo_qtl.py
│   └── treinamento_qtl.py
├── examples/telegrafista_qtl.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/telegrafista_qtl.py
```

---

**© 2026 Luiz Tiago Wilcke**
