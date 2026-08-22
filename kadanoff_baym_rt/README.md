# Equações de Kadanoff–Baym em Tempo Real (RT-KBE) com Auto-Energias \(GW\)/Fock

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Transporte quântico **transiente não-Markoviano** fora do equilíbrio (femtossegundos). Propagação de funções de Green de **dois tempos** no contorno de Keldysh:
\(G^<\), \(G^>\), \(G^R\), com kernels de memória e correlações elétron–elétron (\(GW\)/HF).

---

## 1. Física

| Objeto | Papel |
|--------|--------|
| \(G^<(t_1,t_2)\) | Densidade / correlações menores |
| \(G^R(t_1,t_2)\) | Resposta retardada (causal) |
| \(\Sigma^{HF}\) | Hartree–Fock instantâneo |
| \(\Sigma^{GW}\) | Correlação dinâmica (memória) |

---

## 2. Equações

### Kadanoff–Baym (\(t_1\))

$$
i\hbar\partial_{t_1}G^\lessgtr
=
\bigl[h_0(t_1)+\Sigma^{HF}(t_1)\bigr]G^\lessgtr
+
\int_{t_0}^{t_1}\Sigma^R(t_1,\bar t)\,G^\lessgtr(\bar t,t_2)\,d\bar t
+
\int_{t_0}^{t_2}\Sigma^\lessgtr(t_1,\bar t)\,G^A(\bar t,t_2)\,d\bar t
$$

### Kadanoff–Baym (\(t_2\))

$$
-i\hbar\partial_{t_2}G^\lessgtr
=
G^\lessgtr\bigl[h_0+\Sigma^{HF}\bigr]
+
\int G^R\Sigma^\lessgtr
+
\int G^\lessgtr\Sigma^A
$$

### Auto-energia \(GW\)

$$
\Sigma^\lessgtr=i\hbar\,G^\lessgtr W^\lessgtr
$$

---

## 3. Resíduo PINN

Rede \(G_\theta^<(t_1,t_2)\), \(G_\theta^R(t_1,t_2)\):

$$
\mathcal{L}
=
\bigl\lVert i\hbar\partial_{t_1}G^\lessgtr-[h+\Sigma^{HF}]G^\lessgtr-\!\int(\Sigma^R G^\lessgtr+\Sigma^\lessgtr G^A)\bigr\rVert^2
+
\lambda_{\mathrm{adj}}
\bigl\lVert G^>(t_1,t_2)-[G^<(t_2,t_1)]^\dagger\bigr\rVert^2
+
\lambda_{\mathrm{caus}}
\bigl\lVert G^R\theta(t_2-t_1)\bigr\rVert^2
$$

---

## 4. Estrutura

```
kadanoff_baym_rt/
├── src/
│   ├── fisica_kb.py
│   ├── rede_pinn_kb.py
│   ├── residuo_kb.py
│   └── treinamento_kb.py
├── examples/kadanoff_baym.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/kadanoff_baym.py
```

---

**© 2026 Luiz Tiago Wilcke**
