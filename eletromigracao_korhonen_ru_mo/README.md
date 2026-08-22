# Eletromigração Atômica e Tensão de Acoplamento em Interconexões Metálicas de Ru/Mo

**Autor:** Luiz Tiago Wilcke — Bacharel em Estatística  
**Framework:** PyTorch  

Em nós de **~1 nm**, interconexões **Ru/Mo** operam sob densidades de corrente extremas (\(J>10^7\,\mathrm{A/cm}^2\)), causando deslocamento de átomos pela **força de vento de elétrons** acoplada ao gradiente de tensão mecânica (**equação de Korhonen** modificada).

---

## 1. Física

| Conceito | Descrição |
|----------|-----------|
| **Eletromigração** | Fluxo atômico induzido por corrente |
| **Vento de elétrons** | Força \(\propto Z^* e \rho J\) |
| **Korhonen** | Evolução de \(\sigma_H\) por divergência do fluxo atômico |
| **Blocking boundary** | Fluxo de massa nulo nas vias/contatos |

---

## 2. Equações

### Evolução da tensão hidrostática (Korhonen)

$$
\frac{\partial \sigma_H}{\partial t}
=
\nabla \cdot
\left[
\frac{D_a B \Omega}{k_B T}
\left(
\nabla \sigma_H
-
\frac{Z^* e \rho}{\Omega}\,J
\right)
\right]
$$

### Conservação de corrente

$$
\nabla \cdot (\sigma_{\mathrm{cond}}\nabla\phi)=0,
\qquad
J=-\sigma_{\mathrm{cond}}\nabla\phi
$$

### Forma efetiva 1D

$$
\partial_t\sigma_H
=
D_{\mathrm{eff}}\,
\partial_x
\Bigl(
\partial_x\sigma_H
+
\frac{Z^* e}{\Omega}\,\partial_x\phi
\Bigr)
$$

### Condição de fluxo bloqueado

$$
\left.
\left(
\nabla\sigma_H
+
\frac{Z^* e}{\Omega}\nabla\phi
\right)
\cdot\hat n
\right|_{\text{fronteira}}
=0
$$

---

## 3. Resíduo PINN

$$
\mathcal{L}
=
\left\lVert
\partial_t\sigma_H
-
\nabla\cdot
\Bigl(
D_{\mathrm{eff}}
\bigl[
\nabla\sigma_H
+
\tfrac{Z^* e}{\Omega}\nabla\phi
\bigr]
\Bigr)
\right\rVert^2
+
\lambda_{\mathrm{flux}}
\left\lVert
\bigl(
\nabla\sigma_H
+
\tfrac{Z^* e}{\Omega}\nabla\phi
\bigr)
\cdot\hat n
\big|_{\partial\Omega}
\right\rVert^2
+
\lambda_{\phi}
\lVert\nabla\cdot(\sigma_{\mathrm{cond}}\nabla\phi)\rVert^2
+
\lambda_{\mathrm{IC}}
\lVert\sigma_H(x,0)\rVert^2
$$

---

## 4. Estrutura

```
eletromigracao_korhonen_ru_mo/
├── src/
│   ├── fisica_korhonen.py
│   ├── rede_pinn_em.py       # φ(x), σ_H(x,t)
│   ├── residuo_em.py
│   └── treinamento_em.py
├── examples/em_korhonen.py
├── figures/
├── artigo/
└── README.md
```

---

## 5. Uso

```bash
pip install -r requirements.txt
python examples/em_korhonen.py
```

---

**© 2026 Luiz Tiago Wilcke**
